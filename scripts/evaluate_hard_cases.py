import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.teacher_model import TeacherDeepLOB
from src.models.student_model import StudentMLP
from src.utils.utils import resolve_device

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Trained Networks exclusively on Hard Registered Samples")
    parser.add_argument("--train_path", type=str, default="datasets/FI2010_train.csv")
    parser.add_argument("--registry_path", type=str, default="reports/analysis/hard_cases_registry.json")
    parser.add_argument("--student_weights", type=str, required=True)
    parser.add_argument("--teacher_weights", type=str, required=True)
    parser.add_argument("--window_size", type=int, default=50)
    return parser.parse_args()

def main():
    args = parse_args()
    device = resolve_device("cuda")
    
    df = pd.read_csv(args.train_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    label_col = df.columns[-1]
    feature_cols = [c for c in df.columns if c != label_col][:40]
    features, labels = df[feature_cols].to_numpy(), df[label_col].to_numpy()
    if labels.min() == 1:
        labels = labels - 1
        
    with open(args.registry_path, "r") as f:
        registry = json.load(f)
        
    # Agregacja i automatyczne usuwanie duplikatów z rejestru anomalii
    all_hard_ticks = []
    for asset, lists in registry.items():
        all_hard_ticks.extend(lists["volume_shock_ticks"])
        all_hard_ticks.extend(lists["depth_divergence_ticks"])
        
    # sorted(list(set(...))) gwarantuje nam unikalność i zachowanie ciągłości osi czasu
    unique_hard_indices = sorted(list(set(all_hard_ticks)))
    print(f"Loaded registry file. Total unique expert hard samples processed: {len(unique_hard_indices)}")
    
    # Filtrowanie indeksów pod kątem wielkości okna historycznego (padding)
    valid_eval_indices = [idx for idx in unique_hard_indices if idx >= args.window_size]
    
    windows = [features[idx - args.window_size + 1 : idx + 1] for idx in valid_eval_indices]
    targets = [labels[idx] for idx in valid_eval_indices]
    
    input_tensor = torch.tensor(np.array(windows), dtype=torch.float32).to(device)
    
    teacher = TeacherDeepLOB(num_classes=3).to(device)
    teacher.load_state_dict(torch.load(args.teacher_weights, map_location=device))
    teacher.eval()
    
    student = StudentMLP(window_size=args.window_size, num_features=40, num_classes=3).to(device)
    student.load_state_dict(torch.load(args.student_weights, map_location=device))
    student.eval()
    
    with torch.no_grad():
        t_preds = teacher(input_tensor).max(1)[1].cpu().numpy()
        s_preds = student(input_tensor).max(1)[1].cpu().numpy()
        
    print("\n==========================================================")
    print(" HARD SAMPLES PERFORMANCE REPORT (UNIQUE EXPERT REGIMES)")
    print("==========================================================")
    print(f"Teacher (DeepLOB) Accuracy on Hard Cases: {accuracy_score(targets, t_preds)*100:.2f}%")
    print(f"Student (MLP) Accuracy on Hard Cases:     {accuracy_score(targets, s_preds)*100:.2f}%")
    print("==========================================================\n")
    print("Detailed Student Classification Subspace:")
    print(classification_report(targets, s_preds, target_names=["Up", "Flat", "Down"], zero_division=0))

if __name__ == "__main__":
    main()