import os
import sys
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import yaml
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score

from src.data.dataloader import create_dataloaders
from src.models.baseline import QueueImbalanceBaseline
from src.models.teacher_model import TeacherDeepLOB
from src.models.student_model import StudentMLP
from src.utils import resolve_device

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Model Pipelines vs Baseline Contexts")
    parser.add_argument("--teacher_run", type=str, help="Folder name of the specific teacher run (e.g. run_20260527_113045)")
    parser.add_argument("--student_run", type=str, help="Folder name of the specific student run (e.g. run_20260527_114512)")
    return parser.parse_args()

def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

@torch.no_grad()
def evaluate_ml_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    for x, y in tqdm(dataloader, desc="Evaluating Model", unit="batch", leave=False):
        x = x.to(device)
        outputs = model(x)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(y.numpy())
        
    return all_targets, all_preds

def evaluate_baseline(baseline, dataloader):
    all_preds = []
    all_targets = []
    
    for x, y in tqdm(dataloader, desc="Evaluating Baseline", unit="batch", leave=False):
        predicted = baseline.predict(x)
        all_preds.extend(predicted.numpy())
        all_targets.extend(y.numpy())
        
    return all_targets, all_preds

def generate_performance_text(name, y_true, y_pred):
    report_str = f"PERFORMANCE REPORT: {name}\n"
    report_str += f"Overall Accuracy: {accuracy_score(y_true, y_pred)*100:.2f}%\n\n"
    report_str += classification_report(y_true, y_pred, target_names=["Up", "Flat", "Down"], zero_division=0)
    report_str += "\n"
    return report_str

def main():
    args = parse_args()
    root_path = Path(PROJECT_ROOT)
    main_cfg = load_yaml(root_path / "config" / "config.yaml")
    teacher_cfg = load_yaml(root_path / "config" / "model" / "teacher.yaml")
    student_cfg = load_yaml(root_path / "config" / "model" / "student.yaml")
    data_cfg = load_yaml(root_path / "config" / "dataset" / "fi2010.yaml")
    
    device = resolve_device(main_cfg["device"])
    
    print("Loading test data pipelines...")
    _, _, test_loader = create_dataloaders()
    
    # 1. Evaluate Baseline
    print("\nRunning baseline inference...")
    baseline = QueueImbalanceBaseline(threshold=0.1)
    y_true, y_pred_baseline = evaluate_baseline(baseline, test_loader)
    baseline_report = generate_performance_text("Analytical Baseline (Queue Imbalance)", y_true, y_pred_baseline)
    print(baseline_report)
    
    # 2. Evaluate Teacher
    teacher_report = ""
    if args.teacher_run:
        teacher_path = root_path / teacher_cfg["save_dir"] / args.teacher_run / teacher_cfg["checkpoint_name"]
        if teacher_path.exists():
            print(f"\nLoading Teacher network state from {args.teacher_run} and running inference...")
            teacher = TeacherDeepLOB(num_classes=3).to(device)
            teacher.load_state_dict(torch.load(teacher_path, map_location=device))
            _, y_pred_teacher = evaluate_ml_model(teacher, test_loader, device)
            teacher_report = generate_performance_text(f"Teacher Model (DeepLOB) - {args.teacher_run}", y_true, y_pred_teacher)
            print(teacher_report)
            
            # Save report locally inside the teacher's run folder
            with open(teacher_path.parent / "evaluation_report.txt", "w") as f:
                f.write(teacher_report)
        else:
            print(f"\n[WARNING] Explicit teacher checkpoint file missing at target path: {teacher_path}")

    # 3. Evaluate Student
    student_report = ""
    if args.student_run:
        student_path = root_path / student_cfg["save_dir"] / args.student_run / student_cfg["checkpoint_name"]
        if student_path.exists():
            print(f"\nLoading Student network state from {args.student_run} and running inference...")
            student = StudentMLP(window_size=data_cfg["window_size"], num_features=40, num_classes=3).to(device)
            student.load_state_dict(torch.load(student_path, map_location=device))
            _, y_pred_student = evaluate_ml_model(student, test_loader, device)
            student_report = generate_performance_text(f"Student Model (MLP via Distillation) - {args.student_run}", y_true, y_pred_student)
            print(student_report)
            
            # Save report locally inside the student's run folder
            with open(student_path.parent / "evaluation_report.txt", "w") as f:
                f.write(baseline_report + "\n" + teacher_report + "\n" + student_report)
        else:
            print(f"\n[WARNING] Explicit student checkpoint file missing at target path: {student_path}")

if __name__ == "__main__":
    main()