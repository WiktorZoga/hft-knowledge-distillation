"""Sweep teacher+student training over dataset variants (stocks x horizons).

For every requested combination of stock ("all" or 0..4) and prediction-horizon
index (0..4 = k of 10/20/30/50/100 events) the script:

1. trains a teacher (``scripts/train_teacher.py``),
2. distils a student from it (``scripts/train_student.py``),
3. evaluates baseline / teacher / student on the *test* split of the same
   variant: accuracy, macro F1, per-class F1 and the row-normalised confusion
   matrix.

Results land in ``results/experiments_<sweep_id>/``: one JSON per variant plus
``summary.json`` / ``summary.csv`` / ``summary.md`` for cross-variant analysis.

Example (full grid at the training horizon):
    python scripts/run_experiments.py --stocks all 0 1 2 3 4 --horizons 0
Smoke test:
    python scripts/run_experiments.py --stocks 0 --horizons 0 \
        --teacher-epochs 1 --student-epochs 1 --subset --wandb_mode disabled
"""

import os
import sys
import argparse
import datetime
import json
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
import yaml
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from tqdm import tqdm

from src.data.dataloader import create_dataloaders
from src.models.baseline import QueueImbalanceBaseline
from src.models.student_model import StudentMLP
from src.models.teacher_model import TeacherDeepLOB
from src.utils.utils import resolve_device

CLASS_NAMES = ["up", "flat", "down"]
HORIZON_EVENTS = {0: 10, 1: 20, 2: 30, 3: 50, 4: 100}


def parse_args():
    parser = argparse.ArgumentParser(description="Run stock/horizon experiment grid")
    parser.add_argument("--stocks", nargs="+", default=["all", "0", "1", "2", "3", "4"],
                        help="Dataset variants: 'all' and/or stock indices 0..4")
    parser.add_argument("--horizons", nargs="+", type=int, default=[0], choices=range(5),
                        help="Prediction horizon indices 0..4 (k = 10/20/30/50/100 events)")
    parser.add_argument("--teacher-epochs", type=int, help="Override teacher epochs")
    parser.add_argument("--student-epochs", type=int, help="Override student epochs")
    parser.add_argument("--subset", action="store_true", help="Use the small dev subset")
    parser.add_argument("--wandb_mode", type=str, choices=["online", "offline", "disabled"],
                        help="W&B mode passed through to the training scripts")
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_script(script: str, cli_args: list) -> bool:
    cmd = [sys.executable, os.path.join(PROJECT_ROOT, "scripts", script), *cli_args]
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"[ERROR] {script} exited with code {result.returncode}")
    return result.returncode == 0


@torch.no_grad()
def collect_predictions(predict_fn, dataloader, desc):
    all_true, all_pred = [], []
    for x, y in tqdm(dataloader, desc=desc, unit="batch", leave=False):
        all_pred.append(predict_fn(x))
        all_true.append(y.numpy())
    return np.concatenate(all_true), np.concatenate(all_pred)


def compute_metrics(y_true, y_pred) -> dict:
    per_class = f1_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    cm_norm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2], normalize="true")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_per_class": dict(zip(CLASS_NAMES, per_class.round(4).tolist())),
        "confusion_matrix_normalized": cm_norm.round(4).tolist(),
        "confusion_matrix_counts": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


def evaluate_variant(stock, horizon, teacher_ckpt: Path, student_ckpt: Path,
                     data_overrides: dict, data_cfg: dict, device) -> dict:
    _, _, test_loader = create_dataloaders(overrides=data_overrides)

    baseline = QueueImbalanceBaseline(threshold=0.1)
    y_true, y_pred = collect_predictions(
        lambda x: baseline.predict(x).numpy(), test_loader, "Baseline (test)")
    results = {"baseline": compute_metrics(y_true, y_pred)}

    teacher = TeacherDeepLOB(num_classes=data_cfg["num_classes"]).to(device)
    teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
    teacher.eval()
    y_true, y_pred = collect_predictions(
        lambda x: teacher(x.to(device)).argmax(1).cpu().numpy(), test_loader, "Teacher (test)")
    results["teacher"] = compute_metrics(y_true, y_pred)

    student = StudentMLP(window_size=data_cfg["window_size"],
                         num_features=data_cfg["num_features"],
                         num_classes=data_cfg["num_classes"]).to(device)
    student.load_state_dict(torch.load(student_ckpt, map_location=device))
    student.eval()
    y_true, y_pred = collect_predictions(
        lambda x: student(x.to(device)).argmax(1).cpu().numpy(), test_loader, "Student (test)")
    results["student"] = compute_metrics(y_true, y_pred)

    return {
        "stock": stock,
        "horizon_idx": horizon,
        "horizon_events": HORIZON_EVENTS[horizon],
        "test_samples": int(len(y_true)),
        "class_names": CLASS_NAMES,
        "models": results,
    }


def main():
    args = parse_args()
    root_path = Path(PROJECT_ROOT)
    main_cfg = load_yaml(root_path / "config" / "config.yaml")
    teacher_cfg = load_yaml(root_path / "config" / "model" / "teacher.yaml")
    student_cfg = load_yaml(root_path / "config" / "model" / "student.yaml")
    data_cfg = load_yaml(root_path / "config" / "dataset" / "fi2010.yaml")
    device = resolve_device(main_cfg["device"])

    sweep_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = root_path / "results" / f"experiments_{sweep_id}"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Sweep {sweep_id}: stocks={args.stocks} horizons={args.horizons} "
          f"-> results in {results_dir}")

    all_results, failures = [], []
    for stock in args.stocks:
        for horizon in args.horizons:
            tag = f"stock{stock}_h{horizon}"
            teacher_run = f"sweep_{sweep_id}_teacher_{tag}"
            student_run = f"sweep_{sweep_id}_student_{tag}"

            common = ["--stock", str(stock), "--horizon", str(horizon)]
            if args.subset:
                common.append("--subset")
            if args.wandb_mode:
                common += ["--wandb_mode", args.wandb_mode]

            teacher_args = [*common, "--run_name", teacher_run]
            if args.teacher_epochs:
                teacher_args += ["--epochs", str(args.teacher_epochs)]
            if not run_script("train_teacher.py", teacher_args):
                failures.append(f"{tag}: teacher training failed")
                continue

            student_args = [*common, "--run_name", student_run, "--teacher_run", teacher_run]
            if args.student_epochs:
                student_args += ["--epochs", str(args.student_epochs)]
            if not run_script("train_student.py", student_args):
                failures.append(f"{tag}: student training failed")
                continue

            print(f"\n>>> Evaluating variant {tag} on the test split")
            data_overrides = {"prediction_horizon_idx": horizon,
                              "stock": stock if stock == "all" else int(stock)}
            if args.subset:
                data_overrides["use_subset"] = True
            variant = evaluate_variant(
                stock, horizon,
                teacher_ckpt=root_path / teacher_cfg["save_dir"] / teacher_run / teacher_cfg["checkpoint_name"],
                student_ckpt=root_path / student_cfg["save_dir"] / student_run / student_cfg["checkpoint_name"],
                data_overrides=data_overrides, data_cfg=data_cfg, device=device,
            )
            variant["teacher_run"] = teacher_run
            variant["student_run"] = student_run
            all_results.append(variant)

            with open(results_dir / f"{tag}.json", "w") as f:
                json.dump(variant, f, indent=2)

    with open(results_dir / "summary.json", "w") as f:
        json.dump({"sweep_id": sweep_id, "failures": failures, "variants": all_results}, f, indent=2)

    # Flat table: one row per (variant, model) for quick cross-variant analysis.
    rows = []
    for variant in all_results:
        for model_name, metrics in variant["models"].items():
            rows.append({
                "stock": variant["stock"],
                "horizon_k": variant["horizon_events"],
                "model": model_name,
                "accuracy": round(metrics["accuracy"], 4),
                "f1_macro": round(metrics["f1_macro"], 4),
                **{f"f1_{c}": metrics["f1_per_class"][c] for c in CLASS_NAMES},
                **{f"recall_{c}": metrics["confusion_matrix_normalized"][i][i]
                   for i, c in enumerate(CLASS_NAMES)},
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(results_dir / "summary.csv", index=False)
    try:
        table_text = summary.to_markdown(index=False)
    except ImportError:  # tabulate not installed
        table_text = summary.to_string(index=False)
    (results_dir / "summary.md").write_text(table_text + "\n")

    print("\n================ SWEEP COMPLETE ================")
    print(table_text)
    if failures:
        print("\nFailed variants:")
        for failure in failures:
            print(f"  - {failure}")
    print(f"\nFull results (incl. normalized confusion matrices): {results_dir}")


if __name__ == "__main__":
    main()
