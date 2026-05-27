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

from src.data.dataloader import create_dataloaders
from src.models.teacher_model import TeacherDeepLOB
from src.models.student_model import StudentMLP
from src.losses.distillation_loss import KnowledgeDistillationLoss
from src.utils import resolve_device

def parse_args():
    parser = argparse.ArgumentParser(description="Train Student Model via Distillation")
    parser.add_argument("--config", type=str, default="config/model/student.yaml", help="Path to student YAML config")
    parser.add_argument("--teacher_run", type=str, required=True, help="Specific folder name of the teacher run (e.g. run_20260527_113045)")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--epochs", type=int, help="Override total epochs")
    parser.add_argument("--alpha", type=float, help="Override distillation loss alpha weight")
    parser.add_argument("--temperature", type=float, help="Override logit smoothing temperature")
    return parser.parse_args()

def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def train_student_epoch(student, teacher, dataloader, criterion, optimizer, device, epoch, total_epochs):
    student.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
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
        
        wandb.log({
            "student_train/batch_loss": loss.item(),
            "student_train/batch_acc": predicted.eq(y).sum().item() / y.size(0)
        })
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{(correct/total)*100:.2f}%"})
        
    return running_loss / total, correct / total

@torch.no_grad()
def evaluate_student(student, dataloader, device, epoch, total_epochs):
    student.eval()
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch [{epoch:02d}/{total_epochs}] (Student Val)", unit="batch", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        outputs = student(x)
        
        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
        
        pbar.set_postfix({"acc": f"{(correct/total)*100:.2f}%"})
        
    return correct / total

def main():
    args = parse_args()
    root_path = Path(PROJECT_ROOT)
    
    main_cfg = load_yaml(root_path / "config" / "config.yaml")
    student_cfg = load_yaml(root_path / args.config)
    teacher_cfg = load_yaml(root_path / "config" / "model" / "teacher.yaml")
    distill_cfg = load_yaml(root_path / "config" / "model" / "distillation.yaml")
    
    # CLI Overrides
    if args.lr: student_cfg["lr"] = args.lr
    if args.epochs: student_cfg["epochs"] = args.epochs
    if args.alpha: distill_cfg["alpha"] = args.alpha
    if args.temperature: distill_cfg["temperature"] = args.temperature

    torch.manual_seed(main_cfg["seed"])
    device = resolve_device(main_cfg["device"])
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"student_run_{timestamp}"
    
    run_dir = root_path / student_cfg["save_dir"] / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / student_cfg["checkpoint_name"]
    
    with open(run_dir / "resolved_config.yaml", "w") as f:
        yaml.safe_dump({**main_cfg, **student_cfg, **distill_cfg}, f)
        
    wandb.init(
        project="hft-knowledge-distillation",
        job_type="distillation",
        name=run_name,
        config={**main_cfg, **student_cfg, **distill_cfg}
    )
    
    print("Loading data configurations and streaming pipeline...")
    train_loader, val_loader, _ = create_dataloaders()
    
    print(f"Loading pre-trained Teacher model weights from run sequence: {args.teacher_run}...")
    teacher = TeacherDeepLOB(num_classes=3).to(device)
    teacher_weights_path = root_path / teacher_cfg["save_dir"] / args.teacher_run / teacher_cfg["checkpoint_name"]
    
    if not teacher_weights_path.exists():
        raise FileNotFoundError(f"Missing pre-trained teacher checkpoint at location: {teacher_weights_path}")
        
    teacher.load_state_dict(torch.load(teacher_weights_path, map_location=device))
    teacher.eval()
    
    student = StudentMLP(window_size=main_cfg.get("window_size", 10), num_features=40, num_classes=3).to(device)
    criterion = KnowledgeDistillationLoss(alpha=distill_cfg["alpha"], temperature=distill_cfg["temperature"])
    optimizer = optim.Adam(student.parameters(), lr=student_cfg["lr"], weight_decay=student_cfg["weight_decay"])
    
    best_val_acc = 0.0
    total_epochs = student_cfg["epochs"]
    
    print(f"\nStarting Student Distillation optimization loop on target device: {device}")
    print(f"Target run workspace directory: {run_dir}")
    print("=======================================================================")
    
    for epoch in range(1, total_epochs + 1):
        train_loss, train_acc = train_student_epoch(
            student, teacher, train_loader, criterion, optimizer, device, epoch, total_epochs
        )
        val_acc = evaluate_student(student, val_loader, device, epoch, total_epochs)
        
        print(f"Epoch [{epoch:02d}/{total_epochs}] -> "
              f"Student Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% || "
              f"Student Val Acc: {val_acc*100:.2f}%")
        
        wandb.log({
            "epoch": epoch,
            "student/epoch_loss": train_loss,
            "student/epoch_train_acc": train_acc,
            "student/epoch_val_acc": val_acc
        })
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(student.state_dict(), checkpoint_path)
            print(f" => Saved new optimized student verification checkpoint to: {checkpoint_path}")

    print("\nDISTILLATION PIPELINE COMPLETE")
    wandb.finish()

if __name__ == "__main__":
    main()