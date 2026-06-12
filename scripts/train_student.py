import os
import sys
import datetime
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.optim as optim
import yaml
from pathlib import Path
from tqdm import tqdm
import wandb
from dotenv import load_dotenv
from sklearn.metrics import f1_score

from src.data.dataloader import create_dataloaders
from src.models.teacher_model import TeacherDeepLOB
from src.models.student_model import StudentMLP
from src.losses.distillation_loss import KnowledgeDistillationLoss
from src.utils.utils import log_confusion_matrix, resolve_device, set_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Train Student Model via Distillation")
    parser.add_argument("--config", type=str, default="config/model/student.yaml", help="Path to student YAML config")
    parser.add_argument("--teacher_run", type=str, required=True, help="Specific folder name of the teacher run (e.g. run_20260527_113045)")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--epochs", type=int, help="Override total epochs")
    parser.add_argument("--alpha", type=float, help="Override distillation loss alpha weight")
    parser.add_argument("--temperature", type=float, help="Override logit smoothing temperature")
    parser.add_argument("--stock", type=str, default=None,
                        help="Override dataset stock selection: 'all' or an index 0..4")
    parser.add_argument("--horizon", type=int, default=None, choices=range(5),
                        help="Override prediction horizon index 0..4 (k = 10/20/30/50/100 events)")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Override the run folder / W&B run name (default: student_run_<timestamp>)")
    parser.add_argument("--subset", action="store_true",
                        help="Use the small dev subset regardless of config.yaml")
    parser.add_argument("--wandb_mode", type=str, choices=["online", "offline", "disabled"],
                        help="Override Weights & Biases logging mode")
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

# Label mapping produced by the dataloader: 0 = Up, 1 = Flat, 2 = Down.
CLASS_NAMES = ["up", "flat", "down"]

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

def train_student_epoch(student, teacher, dataloader, criterion, optimizer, device, epoch, total_epochs):
    student.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    pbar = tqdm(dataloader, desc=f"Epoch [{epoch:02d}/{total_epochs}] (Student Train)", unit="batch", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        student_logits = student(x)

        with torch.no_grad():
            teacher_logits = teacher(x)

        loss = criterion(student_logits, teacher_logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        _, predicted = student_logits.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
        all_preds.append(predicted.cpu())
        all_targets.append(y.cpu())

        wandb.log({
            "student_train/batch_loss": loss.item(),
            "student_train/batch_acc": predicted.eq(y).sum().item() / y.size(0)
        })

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{(correct/total)*100:.2f}%"})

    f1_macro, f1_per_class = compute_f1(all_targets, all_preds)
    return running_loss / total, correct / total, f1_macro, f1_per_class

@torch.no_grad()
def evaluate_student(student, dataloader, device, epoch, total_epochs):
    student.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    pbar = tqdm(dataloader, desc=f"Epoch [{epoch:02d}/{total_epochs}] (Student Val)", unit="batch", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        outputs = student(x)

        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
        all_preds.append(predicted.cpu())
        all_targets.append(y.cpu())

        pbar.set_postfix({"acc": f"{(correct/total)*100:.2f}%"})

    f1_macro, f1_per_class = compute_f1(all_targets, all_preds)
    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()
    return correct / total, f1_macro, f1_per_class, y_true, y_pred

def main():
    args = parse_args()
    root_path = Path(PROJECT_ROOT)

    # Load secrets (e.g. WANDB_API_KEY) from .env so logging works non-interactively.
    load_dotenv(root_path / ".env")

    main_cfg = load_yaml(root_path / "config" / "config.yaml")
    student_cfg = load_yaml(root_path / args.config)
    teacher_cfg = load_yaml(root_path / "config" / "model" / "teacher.yaml")
    distill_cfg = load_yaml(root_path / "config" / "model" / "distillation.yaml")
    data_cfg = load_yaml(root_path / "config" / "dataset" / "fi2010.yaml")

    # CLI Overrides
    if args.lr: student_cfg["lr"] = args.lr
    if args.epochs: student_cfg["epochs"] = args.epochs
    if args.alpha: distill_cfg["alpha"] = args.alpha
    if args.temperature: distill_cfg["temperature"] = args.temperature
    data_overrides = build_data_overrides(args)
    data_cfg.update(data_overrides)

    wandb_cfg = main_cfg.get("wandb", {})
    wandb_mode = args.wandb_mode or wandb_cfg.get("mode", "online")

    set_seed(main_cfg["seed"])
    device = resolve_device(main_cfg["device"])

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"student_run_{timestamp}"

    run_dir = root_path / student_cfg["save_dir"] / (args.run_name or f"run_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / student_cfg["checkpoint_name"]

    # Merge every configurable knob so cross-run analysis sees the full picture.
    # data_cfg (window_size, horizon, batch_size, ...) was previously missing.
    # teacher_run is recorded for provenance; we don't splat teacher_cfg because
    # its keys (lr, epochs, ...) would clobber the student's own values.
    resolved_cfg = {**main_cfg, **student_cfg, **distill_cfg, **data_cfg,
                    "teacher_run": args.teacher_run}
    with open(run_dir / "resolved_config.yaml", "w") as f:
        yaml.safe_dump(resolved_cfg, f)

    wandb.init(
        project=wandb_cfg.get("project", "hft-knowledge-distillation"),
        job_type="distillation",
        name=run_name,
        mode=wandb_mode,
        config=resolved_cfg
    )

    print("Loading data configurations and streaming pipeline...")
    print(f"Dataset variant -> stock: {data_cfg.get('stock', 'all')} | "
          f"horizon idx: {data_cfg['prediction_horizon_idx']}")
    train_loader, val_loader, _ = create_dataloaders(overrides=data_overrides)

    print(f"Loading pre-trained Teacher model weights from run sequence: {args.teacher_run}...")
    teacher = TeacherDeepLOB(num_classes=data_cfg["num_classes"]).to(device)
    teacher_weights_path = root_path / teacher_cfg["save_dir"] / args.teacher_run / teacher_cfg["checkpoint_name"]

    if not teacher_weights_path.exists():
        raise FileNotFoundError(f"Missing pre-trained teacher checkpoint at location: {teacher_weights_path}")

    teacher.load_state_dict(torch.load(teacher_weights_path, map_location=device))
    teacher.eval()

    student = StudentMLP(window_size=data_cfg["window_size"], num_features=data_cfg["num_features"],
                         num_classes=data_cfg["num_classes"]).to(device)
    criterion = KnowledgeDistillationLoss(alpha=distill_cfg["alpha"], temperature=distill_cfg["temperature"])
    optimizer = optim.Adam(student.parameters(), lr=student_cfg["lr"], weight_decay=student_cfg["weight_decay"])

    best_val_acc = 0.0
    total_epochs = student_cfg["epochs"]

    print(f"\nStarting Student Distillation optimization loop on target device: {device}")
    print(f"Target run workspace directory: {run_dir}")
    print("=======================================================================")

    for epoch in range(1, total_epochs + 1):
        train_loss, train_acc, train_f1, train_f1_pc = train_student_epoch(
            student, teacher, train_loader, criterion, optimizer, device, epoch, total_epochs
        )
        val_acc, val_f1, val_f1_pc, val_true, val_pred = evaluate_student(student, val_loader, device, epoch, total_epochs)

        print(f"Epoch [{epoch:02d}/{total_epochs}] -> "
              f"Student Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Train F1: {train_f1:.4f} || "
              f"Student Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1:.4f} "
              f"(up {val_f1_pc['up']:.3f} / flat {val_f1_pc['flat']:.3f} / down {val_f1_pc['down']:.3f})")

        log_payload = {
            "epoch": epoch,
            "student/epoch_loss": train_loss,
            "student/epoch_train_acc": train_acc,
            "student/epoch_train_f1_macro": train_f1,
            "student/epoch_val_acc": val_acc,
            "student/epoch_val_f1_macro": val_f1,
        }
        for cls in CLASS_NAMES:
            log_payload[f"student/epoch_train_f1_{cls}"] = train_f1_pc[cls]
            log_payload[f"student/epoch_val_f1_{cls}"] = val_f1_pc[cls]
        wandb.log(log_payload)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(student.state_dict(), checkpoint_path)
            # Keep the logged confusion matrix in sync with the best checkpoint.
            log_confusion_matrix(val_true, val_pred, CLASS_NAMES, key_prefix="student_val")
            print(f" => Saved new optimized student verification checkpoint to: {checkpoint_path}")

    print("\nDISTILLATION PIPELINE COMPLETE")
    wandb.finish()

if __name__ == "__main__":
    main()
