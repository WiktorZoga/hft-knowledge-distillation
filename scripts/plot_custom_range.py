import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.baseline import QueueImbalanceBaseline
from src.models.teacher_model import TeacherDeepLOB
from src.models.student_model import StudentMLP
from src.utils.utils import resolve_device

def parse_args():
    parser = argparse.ArgumentParser(description="Plot Custom Order Book Range with Comprehensive Hard Case Markers")
    parser.add_argument("--data_path", type=str, default="datasets/FI2010_train.csv")
    parser.add_argument("--registry_path", type=str, default="reports/analysis/hard_cases_registry.json")
    parser.add_argument("--student_weights", type=str, required=True)
    parser.add_argument("--teacher_weights", type=str, required=True)
    parser.add_argument("--asset_id", type=int, required=True, choices=[1,2,3,4,5], help="Asset ID chunk to slice")
    parser.add_argument("--left", type=int, required=True, help="Left step index relative to asset timeline")
    parser.add_argument("--right", type=int, required=True, help="Right step index relative to asset timeline")
    parser.add_argument("--window_size", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="reports/analysis")
    return parser.parse_args()

def main():
    args = parse_args()
    device = resolve_device("cuda")
    
    asset_dir = Path(args.output_dir) / f"asset_{args.asset_id}"
    asset_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(args.data_path)
    if "Unnamed: 0" in df.columns: df = df.drop(columns=["Unnamed: 0"])
    label_col = df.columns[-1]
    feature_cols = [c for c in df.columns if c != label_col][:40]
    
    features, labels = df[feature_cols].to_numpy(), df[label_col].to_numpy()
    if labels.min() == 1: labels = labels - 1

    samples_per_asset = len(labels) // 5
    start_asset_pos = (args.asset_id - 1) * samples_per_asset
    
    eval_start = max(0, start_asset_pos + args.left - args.window_size + 1)
    eval_end = start_asset_pos + args.right + 1
    slice_features = features[eval_start:eval_end]
    
    num_samples = len(slice_features) - args.window_size + 1
    if num_samples <= 0:
        print("[ERROR] Selected sequence range window is invalid.")
        return
        
    windows = [slice_features[i : i + args.window_size] for i in range(num_samples)]
    input_tensor = torch.tensor(np.array(windows), dtype=torch.float32)
    
    baseline_preds = QueueImbalanceBaseline(threshold=0.1).predict(input_tensor).numpy()
    
    teacher = TeacherDeepLOB(num_classes=3).to(device)
    teacher.load_state_dict(torch.load(args.teacher_weights, map_location=device))
    teacher.eval()
    
    student = StudentMLP(window_size=args.window_size, num_features=40, num_classes=3).to(device)
    student.load_state_dict(torch.load(args.student_weights, map_location=device))
    student.eval()
    
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        teacher_preds = teacher(input_tensor).max(1)[1].cpu().numpy()
        student_preds = student(input_tensor).max(1)[1].cpu().numpy()

    plot_time = np.arange(args.left, args.right + 1)
    target_labels = labels[start_asset_pos + args.left : start_asset_pos + args.right + 1]
    target_features = features[start_asset_pos + args.left : start_asset_pos + args.right + 1]

    # Pobieranie wszystkich Hard Samples w podanym zakresie z pliku JSON
    shocks_in_range = []
    divs_in_range = []
    if os.path.exists(args.registry_path):
        with open(args.registry_path, "r") as f:
            registry = json.load(f)
        asset_key = f"asset_{args.asset_id}"
        if asset_key in registry:
            global_shocks = registry[asset_key].get("volume_shock_ticks", [])
            global_divs = registry[asset_key].get("depth_divergence_ticks", [])
            
            shocks_in_range = [g_idx - start_asset_pos for g_idx in global_shocks if args.left <= (g_idx - start_asset_pos) <= args.right]
            divs_in_range = [g_idx - start_asset_pos for g_idx in global_divs if args.left <= (g_idx - start_asset_pos) <= args.right]

    fig, axs = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    
    # Panel 1: Price States
    axs[0].plot(plot_time, target_features[:, 0], label="Ask Level 1", color="darkred", alpha=0.7)
    axs[0].plot(plot_time, target_features[:, 2], label="Bid Level 1", color="darkblue", alpha=0.7)
    axs[0].set_title(f"Custom Visualizer -> Asset {args.asset_id} [Ticks: {args.left} to {args.right}]")
    axs[0].set_ylabel("Price States")
    axs[0].grid(True, linestyle="--", alpha=0.3)
    
    # Panel 2: Volume Delta
    vol_delta = target_features[:, 3] - target_features[:, 1]
    axs[1].fill_between(plot_time, vol_delta, step="pre", color="purple", alpha=0.3, label="L1 Vol Imbalance")
    axs[1].set_ylabel("Volume Delta")
    axs[1].grid(True, linestyle="--", alpha=0.3)
    
    # Panel 3: Comprehensive Alignment
    axs[2].plot(plot_time, target_labels, label="GROUND TRUTH", color="black", lw=3.0)
    axs[2].plot(plot_time, teacher_preds, label="TEACHER (DeepLOB)", color="green", linestyle="-.")
    axs[2].plot(plot_time, student_preds, label="STUDENT (MLP)", color="red", linestyle="--")
    axs[2].plot(plot_time, baseline_preds, label="BASELINE (Queue Imb)", color="cyan", linestyle=":")
    
    axs[2].set_yticks([0, 1, 2])
    axs[2].set_yticklabels(["Up", "Flat", "Down"])
    axs[2].invert_yaxis() 
    axs[2].set_ylabel("Execution Decisions")
    axs[2].set_xlabel("Asset Relative Tick Space")
    axs[2].grid(True, linestyle="--", alpha=0.3)
    
    # --- PROCESOWANIE WSZYSTKICH WYKRYTYCH ANOMALII ---
    # Nakładamy markery (kropki) dla każdego punktu z osobna
    for s_tick in shocks_in_range:
        local_idx = s_tick - args.left
        # Kropka na panelu ceny i wolumenu
        axs[0].scatter(s_tick, target_features[local_idx, 0], color="crimson", s=100, zorder=5, marker="o")
        axs[1].scatter(s_tick, vol_delta[local_idx], color="crimson", s=100, zorder=5, marker="o")
        axs[2].scatter(s_tick, target_labels[local_idx], color="crimson", s=100, zorder=5, marker="o")
            
    for d_tick in divs_in_range:
        local_idx = d_tick - args.left
        axs[0].scatter(d_tick, target_features[local_idx, 2], color="darkorange", s=100, zorder=5, marker="X")
        axs[1].scatter(d_tick, vol_delta[local_idx], color="darkorange", s=100, zorder=5, marker="X")
        axs[2].scatter(d_tick, target_labels[local_idx], color="darkorange", s=100, zorder=5, marker="X")

    # Czyszczenie duplikatów w legendach
    axs[0].legend([plt.Line2D([0], [0], color="darkred"), 
                   plt.Line2D([0], [0], color="darkblue"),
                   plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="crimson", markersize=10),
                   plt.Line2D([0], [0], marker="X", color="none", markerfacecolor="darkorange", markersize=10)],
                  ["Ask L1", "Bid L1", "Volume Shock (Anomalia)", "Depth Divergence (Anomalia)"], loc="upper left")
    
    axs[1].legend(loc="upper left")
    axs[2].legend(loc="upper left", ncol=4)

    filename = asset_dir / f"custom_range_{args.left}_{args.right}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Custom range plot generated successfully. Total markers placed: {len(shocks_in_range) + len(divs_in_range)}")

if __name__ == "__main__":
    main()