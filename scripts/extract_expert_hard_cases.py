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
from src.utils.hardness_filters import check_volume_shock, check_depth_divergence

def parse_args():
    parser = argparse.ArgumentParser(description="Extract Expert Hard Cases to JSON")
    parser.add_argument("--train_path", type=str, default="datasets/FI2010_train.csv")
    parser.add_argument("--output_dir", type=str, default="reports/analysis")
    return parser.parse_args()

def save_dashboard(features, labels, baseline_preds, start_idx, end_idx, asset_id, case_type, save_path):
    time_axis = np.arange(start_idx, end_idx)
    fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    
    # Wyznaczamy centralny punkt okna (tam gdzie aktywował się filtr)
    center_idx = start_idx + (end_idx - start_idx) // 2
    
    # 1. Panel ceny (Mid Price)
    mid_prices = (features[time_axis, 0] + features[time_axis, 2]) / 2.0
    axs[0].plot(time_axis, mid_prices, label="Mid Price", color="black", lw=2)
    
    # RYSOWANIE KROPKI NA CENIE:
    center_mid_price = (features[center_idx, 0] + features[center_idx, 2]) / 2.0
    axs[0].scatter(center_idx, center_mid_price, color="crimson", s=150, zorder=5, 
                    label=f"HARD SAMPLE ANOMALY ({case_type})")
    
    axs[0].set_title(f"Asset {asset_id} - Expert Sample Profile")
    axs[0].set_ylabel("Normalized Price")
    axs[0].legend(loc="upper left")
    axs[0].grid(True, linestyle="--", alpha=0.3)
    
    # 2. Panel wolumenu (Volume Delta)
    vol_delta = features[time_axis, 3] - features[time_axis, 1]
    axs[1].fill_between(time_axis, vol_delta, step="pre", color="purple", alpha=0.4, label="L1 Vol Imbalance")
    
    # RYSOWANIE KROPKI NA WOLUMENIE:
    center_vol_delta = features[center_idx, 3] - features[center_idx, 1]
    axs[1].scatter(center_idx, center_vol_delta, color="crimson", s=150, zorder=5)
    
    axs[1].set_ylabel("Volume Delta")
    axs[1].legend(loc="upper left")
    axs[1].grid(True, linestyle="--", alpha=0.3)
    
    # 3. Panel sygnałów (Ground Truth vs Baseline)
    axs[2].plot(time_axis, labels[time_axis], label="GROUND TRUTH (Market Move)", color="darkgreen", lw=2.5)
    axs[2].plot(time_axis, baseline_preds[time_axis], label="BASELINE (Queue Imb)", color="cyan", linestyle=":", alpha=0.8, lw=1.8)
    
    # RYSOWANIE KROPKI NA SYGNALE RYNKOWYM:
    center_label = labels[center_idx]
    axs[2].scatter(center_idx, center_label, color="crimson", s=150, zorder=5, label="Anomalous Tick")
    
    axs[2].set_yticks([0, 1, 2])
    axs[2].set_yticklabels(["Up", "Flat", "Down"])
    axs[2].invert_yaxis()  # Up na górze, Down na dole
    axs[2].set_ylabel("Signal Label")
    axs[2].set_xlabel("Tick (Sequence Space)")
    axs[2].legend(loc="upper left")
    axs[2].grid(True, linestyle="--", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def main():
    args = parse_args()
    base_out = Path(args.output_dir)
    base_out.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(args.train_path)
    if "Unnamed: 0" in df.columns: df = df.drop(columns=["Unnamed: 0"])
    
    label_col = df.columns[-1]
    feature_cols = [c for c in df.columns if c != label_col][:40]
    features, labels = df[feature_cols].to_numpy(), df[label_col].to_numpy()
    if labels.min() == 1: labels = labels - 1
        
    dummy_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(1)
    baseline_preds = QueueImbalanceBaseline(threshold=0.1).predict(dummy_tensor).numpy()

    samples_per_asset = len(labels) // 5
    registry = {}

    for asset_id in range(1, 6):
        start_pos = (asset_id - 1) * samples_per_asset
        end_pos = asset_id * samples_per_asset
        
        a_features = features[start_pos:end_pos]
        a_labels = labels[start_pos:end_pos]
        a_baseline = baseline_preds[start_pos:end_pos]
        
        asset_dir = base_out / f"asset_{asset_id}"
        asset_dir.mkdir(parents=True, exist_ok=True)
        
        shocks, divergences = [], []
        for i in range(100, len(a_labels) - 100):
            if check_volume_shock(a_features, a_labels, i): shocks.append(int(i + start_pos))
            if check_depth_divergence(a_features, a_labels, i): divergences.append(int(i + start_pos))
                
        registry[f"asset_{asset_id}"] = {"volume_shock_ticks": shocks, "depth_divergence_ticks": divergences}
        
        if len(shocks) > 0:
            l_idx = shocks[len(shocks)//2] - start_pos
            save_dashboard(a_features, a_labels, a_baseline, l_idx-100, l_idx+100, asset_id, "Volume Shock", asset_dir / "case_1_volume_shock.png")
        if len(divergences) > 0:
            l_idx = divergences[len(divergences)//2] - start_pos
            save_dashboard(a_features, a_labels, a_baseline, l_idx-100, l_idx+100, asset_id, "Depth Divergence", asset_dir / "case_2_depth_divergence.png")

    with open(base_out / "hard_cases_registry.json", "w") as f:
        json.dump(registry, f, indent=4)
    print(f"Registry saved successfully: {base_out / 'hard_cases_registry.json'}")

if __name__ == "__main__":
    main()