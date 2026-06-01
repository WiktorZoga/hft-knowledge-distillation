import os
import sys
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.optim as optim
import yaml
import optuna
import wandb
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score

from src.data.dataloader import create_dataloaders
from src.models.teacher_model import TeacherDeepLOB
from src.models.student_model import StudentMLP
from src.losses.distillation_loss import KnowledgeDistillationLoss
from src.utils import resolve_device, set_seed

def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Tuning for Student Distillation")
    parser.add_argument("--teacher_run", type=str, required=True, help="Teacher run folder name to use as frozen backbone")
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials to execute")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per trial")
    return parser.parse_args()

def train_one_epoch(student, teacher, loader, criterion, optimizer, device):
    student.train()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        student_logits = student(x)
        with torch.no_grad():
            teacher_logits = teacher(x)
        loss = criterion(student_logits, teacher_logits, y)
        loss.backward()
        optimizer.step()
        _, predicted = student_logits.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
    return correct / total

@torch.no_grad()
def evaluate(student, loader, device):
    student.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        outputs = student(x)
        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
        all_preds.append(predicted.cpu())
        all_targets.append(y.cpu())
    # Macro F1 so the dominant 'Flat' class can't mask poor Up/Down recall.
    f1_macro = f1_score(torch.cat(all_targets).numpy(), torch.cat(all_preds).numpy(),
                        average="macro", zero_division=0)
    return correct / total, f1_macro

def objective(trial, teacher, train_loader, val_loader, data_cfg, wandb_cfg, wandb_mode, device, epochs, seed, teacher_run):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    alpha = trial.suggest_float("alpha", 0.3, 0.9)
    temperature = trial.suggest_float("temperature", 2.0, 8.0)

    run = wandb.init(
        project=wandb_cfg.get("project", "hft-knowledge-distillation"),
        job_type="optuna-trial",
        name=f"trial_{trial.number}",
        mode=wandb_mode,
        # Log the searched hyperparameters AND the fixed context (dataset knobs,
        # seed, teacher provenance) so each trial is self-describing for later
        # cross-run analysis.
        config={
            "lr": lr,
            "weight_decay": weight_decay,
            "alpha": alpha,
            "temperature": temperature,
            "epochs": epochs,
            "trial": trial.number,
            "seed": seed,
            "teacher_run": teacher_run,
            **data_cfg,
        },
        reinit=True,
    )

    student = StudentMLP(
        window_size=data_cfg["window_size"],
        num_features=data_cfg["num_features"],
        num_classes=data_cfg["num_classes"],
    ).to(device)

    criterion = KnowledgeDistillationLoss(alpha=alpha, temperature=temperature)
    optimizer = optim.Adam(student.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_acc = train_one_epoch(student, teacher, train_loader, criterion, optimizer, device)
        val_acc, val_f1 = evaluate(student, val_loader, device)

        wandb.log({"epoch": epoch, "train_acc": train_acc, "val_acc": val_acc,
                   "val_f1_macro": val_f1})

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        trial.report(val_acc, epoch)
        if trial.should_prune():
            run.finish()
            raise optuna.exceptions.TrialPruned()

    run.finish()
    return best_val_acc

def main():
    args = parse_args()
    root_path = Path(PROJECT_ROOT)

    # Load secrets (e.g. WANDB_API_KEY) from .env so logging works non-interactively.
    load_dotenv(root_path / ".env")

    main_cfg = load_yaml(root_path / "config" / "config.yaml")
    teacher_cfg = load_yaml(root_path / "config" / "model" / "teacher.yaml")
    data_cfg = load_yaml(root_path / "config" / "dataset" / "fi2010.yaml")

    wandb_cfg = main_cfg.get("wandb", {})
    wandb_mode = wandb_cfg.get("mode", "online")

    set_seed(main_cfg["seed"])
    device = resolve_device(main_cfg["device"])

    print("Loading data pipelines...")
    train_loader, val_loader, _ = create_dataloaders()

    print(f"Loading teacher weights from: {args.teacher_run}")
    teacher = TeacherDeepLOB(num_classes=data_cfg["num_classes"]).to(device)
    teacher_path = root_path / teacher_cfg["save_dir"] / args.teacher_run / teacher_cfg["checkpoint_name"]
    if not teacher_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_path}")
    teacher.load_state_dict(torch.load(teacher_path, map_location=device))
    teacher.eval()

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        study_name="student-distillation-tuning",
    )

    study.optimize(
        lambda trial: objective(trial, teacher, train_loader, val_loader, data_cfg, wandb_cfg, wandb_mode, device, args.epochs, main_cfg["seed"], args.teacher_run),
        n_trials=args.trials,
        show_progress_bar=True,
    )

    print("\nTUNING COMPLETE")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best val accuracy: {study.best_value*100:.2f}%")
    print("Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
