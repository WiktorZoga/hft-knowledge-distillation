import os
import sys
import datetime
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from pathlib import Path
from tqdm import tqdm
import wandb
from dotenv import load_dotenv
from sklearn.metrics import f1_score

from src.data.dataloader import create_dataloaders
from src.models.teacher_model import TeacherDeepLOB
from src.utils.eval_benchmark import collect_teacher_latency_metrics
from src.utils.hardness_metrics import compute_hardness_metrics, count_hardness_samples
from src.utils.utils import log_confusion_matrix, resolve_device, set_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Train Teacher Model (DeepLOB)")
    parser.add_argument("--config", type=str, default="config/model/teacher.yaml", help="Path to teacher YAML config")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--weight_decay", type=float, help="Override weight decay")
    parser.add_argument("--epochs", type=int, help="Override total epochs")
    parser.add_argument("--stock", type=str, default=None,
                        help="Override dataset stock selection: 'all' or an index 0..4")
    parser.add_argument("--horizon", type=int, default=None, choices=range(5),
                        help="Override prediction horizon index 0..4 (k = 10/20/30/50/100 events)")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Override the run folder / W&B run name (default: teacher_run_<timestamp>)")
    parser.add_argument("--subset", action="store_true",
                        help="Use the small dev subset regardless of config.yaml")
    parser.add_argument("--wandb_mode", type=str, choices=["online", "offline", "disabled"],
                        help="Override Weights & Biases logging mode")
    parser.add_argument("--wandb_project", type=str, help="Override W&B project name")
    parser.add_argument("--wandb_entity", type=str, help="Override W&B entity (username or team)")
    return parser.parse_args()


def build_data_overrides(args) -> dict:
    """Dataset-config overrides shared by the sweep CLI flags."""
    overrides = {}
    if args.stock is not None:
        overrides["stock"] = args.stock if args.stock == "all" else int(args.stock)
    if args.horizon is not None:
        overrides["prediction_horizon_idx"] = args.horizon
    if args.subset:
        overrides["use_subset"] = True
    return overrides

def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    all_hard = []

    pbar = tqdm(dataloader, desc=f"Epoch [{epoch:02d}/{total_epochs}] (Train)", unit="batch", leave=False)
    for x, y, hard in pbar:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
        all_preds.append(predicted.cpu())
        all_targets.append(y.cpu())
        all_hard.append(hard.cpu())

        wandb.log({
            "train/batch_loss": loss.item(),
            "train/batch_acc": predicted.eq(y).sum().item() / y.size(0)
        })

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{(correct/total)*100:.2f}%"})

    f1_macro, f1_per_class = compute_f1(all_targets, all_preds)
    hard_metrics = compute_hardness_metrics(all_targets, all_preds, all_hard, "train")
    return running_loss / total, correct / total, f1_macro, f1_per_class, hard_metrics

@torch.no_grad()
def evaluate(
    model,
    dataloader,
    criterion,
    device,
    key_prefix: str = "val",
    epoch: int | None = None,
    total_epochs: int | None = None,
    desc: str | None = None,
):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    all_hard = []

    if desc is None:
        desc = (
            f"Epoch [{epoch:02d}/{total_epochs}] (Val)"
            if epoch is not None and total_epochs is not None
            else "Test"
        )

    pbar = tqdm(dataloader, desc=desc, unit="batch", leave=False)
    for x, y, hard in pbar:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        loss = criterion(outputs, y)

        running_loss += loss.item() * x.size(0)
        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
        all_preds.append(predicted.cpu())
        all_targets.append(y.cpu())
        all_hard.append(hard.cpu())

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{(correct/total)*100:.2f}%"})

    f1_macro, f1_per_class = compute_f1(all_targets, all_preds)
    hard_metrics = compute_hardness_metrics(all_targets, all_preds, all_hard, key_prefix)
    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()
    return running_loss / total, correct / total, f1_macro, f1_per_class, hard_metrics, y_true, y_pred

# Label mapping produced by the dataloader: 0 = Up, 1 = Flat, 2 = Down.
CLASS_NAMES = ["up", "flat", "down"]

def compute_class_weights(dataset, num_classes, device):
    """Inverse-frequency class weights over the *windowed* training targets.

    Mirrors sklearn's 'balanced' scheme: ``w_c = N / (num_classes * count_c)``.
    Computed only over the labels actually consumed by ``__getitem__`` (the
    horizon column, offset by ``window_size - 1``) so the weights match what the
    model is trained against rather than the raw label pool. Counters the heavy
    'flat' majority in FI2010 so Up/Down stop being drowned out. Handles both a
    plain FI2010Dataset and the per-stock ConcatDataset built by the dataloader.
    """
    sub_datasets = getattr(dataset, "datasets", [dataset])
    targets = np.concatenate([d.windowed_targets() for d in sub_datasets])
    counts = np.bincount(targets, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0  # avoid div-by-zero for an absent class
    weights = targets.shape[0] / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def compute_f1(targets, preds):
    """Macro F1 plus per-class F1 over the collected per-batch prediction tensors.

    Macro (not weighted) so the dominant 'Flat' class can't mask poor Up/Down
    recall — the metric that actually matters for LOB direction. Per-class F1
    exposes exactly which direction the model is missing. `labels=[0,1,2]` pins
    the order and keeps a class present even if absent from a given split.
    """
    y_true = torch.cat(targets).numpy()
    y_pred = torch.cat(preds).numpy()
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class = f1_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    return macro, dict(zip(CLASS_NAMES, per_class))

def main():
    args = parse_args()
    root_path = Path(PROJECT_ROOT)

    # Load secrets (e.g. WANDB_API_KEY) from .env so logging works non-interactively.
    load_dotenv(root_path / ".env")

    main_cfg = load_yaml(root_path / "config" / "config.yaml")
    teacher_cfg = load_yaml(root_path / args.config)
    # Dataset config (window_size, horizon, batch_size, ...) is consumed inside
    # create_dataloaders(); load it here too so every configurable knob lands in
    # the W&B run config and the on-disk resolved snapshot for later analysis.
    data_cfg = load_yaml(root_path / "config" / "dataset" / "fi2010.yaml")

    # CLI Overrides
    if args.lr: teacher_cfg["lr"] = args.lr
    if args.weight_decay: teacher_cfg["weight_decay"] = args.weight_decay
    if args.epochs: teacher_cfg["epochs"] = args.epochs
    data_overrides = build_data_overrides(args)
    data_cfg.update(data_overrides)

    wandb_cfg = main_cfg.get("wandb", {})
    wandb_mode = args.wandb_mode or wandb_cfg.get("mode", "online")
    wandb_project = args.wandb_project or wandb_cfg.get("project", "hft-knowledge-distillation")
    wandb_entity = args.wandb_entity or wandb_cfg.get("entity", None)

    set_seed(main_cfg["seed"])
    device = resolve_device(main_cfg["device"])

    # Generate timestamped run identifier
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"teacher_run_{timestamp}"

    run_dir = root_path / teacher_cfg["save_dir"] / (args.run_name or f"run_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / teacher_cfg["checkpoint_name"]
    
    # Dump active configuration profile to the run folder
    resolved_cfg = {**main_cfg, **teacher_cfg, **data_cfg}
    with open(run_dir / "resolved_config.yaml", "w") as f:
        yaml.safe_dump(resolved_cfg, f)

    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        job_type="teacher-training",
        name=run_name,
        mode=wandb_mode,
        config=resolved_cfg
    )
    
    print("Loading data configurations and streaming pipeline...")
    print(f"Dataset variant -> stock: {data_cfg.get('stock', 'all')} | "
          f"horizon idx: {data_cfg['prediction_horizon_idx']}")
    train_loader, val_loader, test_loader = create_dataloaders(overrides=data_overrides)

    train_hard_stats = count_hardness_samples(train_loader.dataset)
    val_hard_stats = count_hardness_samples(val_loader.dataset)
    test_hard_stats = count_hardness_samples(test_loader.dataset)
    wandb.config.update({
        "dataset/train_hard_windows": train_hard_stats,
        "dataset/val_hard_windows": val_hard_stats,
        "dataset/test_hard_windows": test_hard_stats,
    })
    print(f"Hard windows -> train: {train_hard_stats} | val: {val_hard_stats} | test: {test_hard_stats}")

    model = TeacherDeepLOB(num_classes=3).to(device)

    # Class-imbalance handling: weight CE by inverse training-class frequency.
    if teacher_cfg.get("class_weights", True):
        class_weights = compute_class_weights(train_loader.dataset, num_classes=3, device=device)
        print(f"Using inverse-frequency class weights (up/flat/down): "
              f"{class_weights.cpu().numpy().round(3).tolist()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(), 
        lr=teacher_cfg["lr"], 
        weight_decay=teacher_cfg["weight_decay"]
    )
    
    total_epochs = teacher_cfg["epochs"]

    # Early stopping: monitor a validation metric (higher = better) and stop once
    # it has not improved for `patience` consecutive epochs. The best-so-far
    # checkpoint is kept on disk.
    monitor = teacher_cfg.get("early_stopping_metric", "val_f1_macro")
    patience = teacher_cfg.get("early_stopping_patience", 8)
    best_metric = float("-inf")
    epochs_no_improve = 0

    print(f"\nStarting Teacher model execution loop on target device: {device}")
    print(f"Target run workspace directory: {run_dir}")
    print(f"Early stopping on '{monitor}' with patience={patience}")

    for epoch in range(1, total_epochs + 1):
        train_loss, train_acc, train_f1, train_f1_pc, train_hard = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, total_epochs
        )
        val_loss, val_acc, val_f1, val_f1_pc, val_hard, val_true, val_pred = evaluate(
            model, val_loader, criterion, device, key_prefix="val",
            epoch=epoch, total_epochs=total_epochs,
        )

        print(f"Epoch [{epoch:02d}/{total_epochs}] -> "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Train F1: {train_f1:.4f} || "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1:.4f} "
              f"(up {val_f1_pc['up']:.3f} / flat {val_f1_pc['flat']:.3f} / down {val_f1_pc['down']:.3f})")

        log_payload = {
            "epoch": epoch,
            "train/epoch_loss": train_loss,
            "train/epoch_acc": train_acc,
            "train/epoch_f1_macro": train_f1,
            "val/epoch_loss": val_loss,
            "val/epoch_acc": val_acc,
            "val/epoch_f1_macro": val_f1,
        }
        for cls in CLASS_NAMES:
            log_payload[f"train/epoch_f1_{cls}"] = train_f1_pc[cls]
            log_payload[f"val/epoch_f1_{cls}"] = val_f1_pc[cls]
        log_payload.update(train_hard)
        log_payload.update(val_hard)
        wandb.log(log_payload)

        current = val_f1 if monitor == "val_f1_macro" else val_acc
        if current > best_metric:
            best_metric = current
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            # Keep the logged confusion matrix in sync with the best checkpoint.
            log_confusion_matrix(val_true, val_pred, CLASS_NAMES, key_prefix="val")
            print(f" => New best {monitor}={current:.4f}; saved checkpoint to: {checkpoint_path}")
        else:
            epochs_no_improve += 1
            print(f" => No improvement on {monitor} for {epochs_no_improve}/{patience} epoch(s) "
                  f"(best={best_metric:.4f})")
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    wandb.run.summary["best/" + monitor] = best_metric

    print("\nEvaluating best checkpoint on the test split...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_loss, test_acc, test_f1, test_f1_pc, test_hard, test_true, test_pred = evaluate(
        model, test_loader, criterion, device, key_prefix="test", desc="Test",
    )
    log_confusion_matrix(test_true, test_pred, CLASS_NAMES, key_prefix="test")
    latency_metrics = collect_teacher_latency_metrics(model, test_loader, device)
    final_payload = {
        "test/epoch_loss": test_loss,
        "test/epoch_acc": test_acc,
        "test/epoch_f1_macro": test_f1,
        **{f"test/epoch_f1_{cls}": test_f1_pc[cls] for cls in CLASS_NAMES},
        **test_hard,
        **latency_metrics,
    }
    wandb.log(final_payload)
    wandb.run.summary.update(final_payload)
    print(
        f"Test -> Loss: {test_loss:.4f} | Acc: {test_acc * 100:.2f}% | F1: {test_f1:.4f} "
        f"(up {test_f1_pc['up']:.3f} / flat {test_f1_pc['flat']:.3f} / down {test_f1_pc['down']:.3f}) | "
        f"Teacher mean latency: {latency_metrics['latency/teacher_mean_us']:.1f} µs"
    )

    print("\nOPTIMIZATION PIPELINE COMPLETE")
    wandb.finish()

if __name__ == "__main__":
    main()