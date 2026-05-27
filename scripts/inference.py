import os
import sys
import argparse
import numpy as np
import torch
import yaml
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.baseline import QueueImbalanceBaseline
from src.models.teacher_model import TeacherDeepLOB
from src.models.student_model import StudentMLP
from src.utils import resolve_device

def parse_args():
    parser = argparse.ArgumentParser(description="Production Inference Script for LOB Models")
    parser.add_argument("--model_type", type=str, required=True, choices=["baseline", "teacher", "student"], help="Type of model backend to execute")
    parser.add_argument("--weights", type=str, help="Path to the specific trained .pt model weights file (not required for baseline)")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the target raw data .txt file to run inference on")
    parser.add_argument("--window_size", type=int, default=10, help="Historical sequence window size")
    parser.add_argument("--threshold", type=float, default=0.1, help="Queue Imbalance threshold (baseline only)")
    return parser.parse_args()

def load_and_preprocess_file(file_path: str, window_size: int) -> torch.Tensor:
    """
    Loads a single raw text file and prepares sliding windows for direct execution.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Target data file not found at: {file_path}")
        
    # Load raw text matrix (transposed to match features dimension context)
    data = np.loadtxt(file_path)
    data = data.T
    features = data[:, :40]
    
    # Apply standard normalization mapping in-place for this specific file scope
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    std[std == 0] = 1.0
    features = (features - mean) / std
    
    # Construct historical sliding window sequences
    num_samples = len(features) - window_size + 1
    if num_samples <= 0:
        raise ValueError(f"Data file too short for the requested window size of {window_size}")
        
    windows = []
    for idx in range(num_samples):
        windows.append(features[idx : idx + window_size])
        
    return torch.tensor(np.array(windows), dtype=torch.float32)

def main():
    args = parse_args()

    with open(Path(PROJECT_ROOT) / "config" / "config.yaml", "r") as f:
        main_cfg = yaml.safe_load(f)
    device = resolve_device(main_cfg["device"])
    
    print(f"Preprocessing target data file: {args.data_path}")
    try:
        input_tensor = load_and_preprocess_file(args.data_path, args.window_size)
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return

    print(f"Total processed sequences ready for inference: {input_tensor.size(0)}")
    print(f"Initializing model framework backend: {args.model_type.upper()}")
    
    # 1. Pipeline Routing: Analytical Baseline Execution
    if args.model_type == "baseline":
        model = QueueImbalanceBaseline(threshold=args.threshold)
        # Move inputs to CPU since baseline runs native PyTorch tensors locally
        preds = model.predict(input_tensor)
        
    # 2. Pipeline Routing: Neural Network Branches (Teacher / Student)
    else:
        if not args.weights:
            print("[ERROR] Neural network inference requires a valid path via the '--weights' flag.")
            return
            
        if args.model_type == "teacher":
            model = TeacherDeepLOB(num_classes=3)
        else: # student
            model = StudentMLP(window_size=args.window_size, num_features=40, num_classes=3)
            
        print(f"Loading state weights checkpoint from: {args.weights}")
        try:
            model.load_state_dict(torch.load(args.weights, map_location=device))
        except Exception as e:
            print(f"[ERROR] Failed to apply target weights state: {e}")
            return
            
        model = model.to(device)
        model.eval()
        
        # Stream evaluation contexts through the network
        input_tensor = input_tensor.to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            _, preds = outputs.max(1)
            
    # 3. Output Aggregation Summary
    preds_np = preds.cpu().numpy()
    total_signals = len(preds_np)
    
    unique, counts = np.unique(preds_np, return_counts=True)
    distribution = dict(zip(unique, counts))
    
    print("INFERENCE EXECUTION SUMMARY\n")
    print(f"Backend Engine:        {args.model_type.upper()}")
    print(f"Total Predicted Ticks: {total_signals}")
    print(f"Signals Distribution:  Up (0): {distribution.get(0, 0)} | Flat (1): {distribution.get(1, 0)} | Down (2): {distribution.get(2, 0)}\n")
    print("Inference lifecycle terminated successfully.")

if __name__ == "__main__":
    main()