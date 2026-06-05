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
from src.utils.utils import resolve_device, set_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Train Teacher Model (DeepLOB)")
    parser.add_argument("--config", type=str, default="config/model/teacher.yaml", help="Path to teacher YAML config")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--weight_decay", type=float, help="Override weight decay")
    parser.add_argument("--epochs", type=int, help="Override total epochs")
    parser.add_argument("--wandb_mode", type=str, choices=["online", "offline", "disabled"],
                        help="Override Weights & Biases logging mode")
    return parser.parse_args()

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

    pbar = tqdm(dataloader, desc=f"Epoch [{epoch:02d}/{total_epochs}] (Train)", unit="batch", leave=False)
    for x, y in pbar:
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

        wandb.log({
            "train/batch_loss": loss.item(),
            "train/batch_acc": predicted.eq(y).sum().item() / y.size(0)
        })

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{(correct/total)*100:.2f}%"})

    f1_macro, f1_per_class = compute_f1(all_targets, all_preds)
    return running_loss / total, correct / total, f1_macro, f1_per_class

@torch.no_grad()
def evaluate(model, dataloader, criterion, device, epoch, total_epochs):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    pbar = tqdm(dataloader, desc=f"Epoch [{epoch:02d}/{total_epochs}] (Val)", unit="batch", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        loss = criterion(outputs, y)

        running_loss += loss.item() * x.size(0)
        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
        all_preds.append(predicted.cpu())
        all_targets.append(y.cpu())

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{(correct/total)*100:.2f}%"})

    f1_macro, f1_per_class = compute_f1(all_targets, all_preds)
    return running_loss / total, correct / total, f1_macro, f1_per_class

# Label mapping produced by the dataloader: 0 = Up, 1 = Flat, 2 = Down.
CLASS_NAMES = ["up", "flat", "down"]

def compute_class_weights(dataset, num_classes, device):
    """Inverse-frequency class weights over the *windowed* training targets.

    Mirrors sklearn's 'balanced' scheme: ``w_c = N / (num_classes * count_c)``.
    Computed only over the labels actually consumed by ``__getitem__`` (the
    horizon column, offset by ``window_size - 1``) so the weights match what the
    model is trained against rather than the raw label pool. Counters the heavy
    'flat' majority in FI2010 so Up/Down stop being drowned out.
    """
    start = dataset.window_size - 1
    targets = dataset.labels[start: start + dataset.num_samples, dataset.prediction_horizon_idx]
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

    wandb_cfg = main_cfg.get("wandb", {})
    wandb_mode = args.wandb_mode or wandb_cfg.get("mode", "online")

    set_seed(main_cfg["seed"])
    device = resolve_device(main_cfg["device"])
    
    # Generate timestamped run identifier
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"teacher_run_{timestamp}"
    
    run_dir = root_path / teacher_cfg["save_dir"] / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / teacher_cfg["checkpoint_name"]
    
    # Dump active configuration profile to the run folder
    resolved_cfg = {**main_cfg, **teacher_cfg, **data_cfg}
    with open(run_dir / "resolved_config.yaml", "w") as f:
        yaml.safe_dump(resolved_cfg, f)

    wandb.init(
        project=wandb_cfg.get("project", "hft-knowledge-distillation"),
        job_type="teacher-training",
        name=run_name,
        mode=wandb_mode,
        config=resolved_cfg
    )
    
    print("Loading data configurations and streaming pipeline...")
    train_loader, val_loader, _ = create_dataloaders()
    
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
        train_loss, train_acc, train_f1, train_f1_pc = train_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs)
        val_loss, val_acc, val_f1, val_f1_pc = evaluate(model, val_loader, criterion, device, epoch, total_epochs)

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
        wandb.log(log_payload)

        current = val_f1 if monitor == "val_f1_macro" else val_acc
        if current > best_metric:
            best_metric = current
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f" => New best {monitor}={current:.4f}; saved checkpoint to: {checkpoint_path}")
        else:
            epochs_no_improve += 1
            print(f" => No improvement on {monitor} for {epochs_no_improve}/{patience} epoch(s) "
                  f"(best={best_metric:.4f})")
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    wandb.run.summary["best/" + monitor] = best_metric

    print("\nOPTIMIZATION PIPELINE COMPLETE")
    wandb.finish()

if __name__ == "__main__":
    main()