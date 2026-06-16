"""Bottom-panel signal alignment plot for W&B (GT vs teacher vs student)."""

from __future__ import annotations

import io
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.dataset import FI2010Dataset


@dataclass(frozen=True)
class SignalTraceWindow:
    """A contiguous validation slice for the signal-trace plot."""

    stock_index: int
    win_start: int
    win_end: int
    tick_start: int
    tick_end: int


def get_val_dataset_for_stock(val_loader_dataset, stock_index: int) -> FI2010Dataset:
    """Return the validation FI2010Dataset for ``stock_index`` from a ConcatDataset."""
    sub_datasets = getattr(val_loader_dataset, "datasets", [val_loader_dataset])
    for ds in sub_datasets:
        if getattr(ds, "stock_index", None) == stock_index:
            return ds
    raise ValueError(
        f"Stock {stock_index} not found in validation dataset. "
        f"Available: {[getattr(d, 'stock_index', None) for d in sub_datasets]}"
    )


def _window_to_ticks(dataset: FI2010Dataset, win_start: int, win_end: int) -> tuple[int, int]:
    offset = dataset.timeline_offset
    ws = dataset.window_size
    tick_start = offset + win_start + ws - 1
    tick_end = offset + win_end + ws - 1
    return tick_start, tick_end


def _ticks_to_window(dataset: FI2010Dataset, tick_start: int, tick_end: int) -> tuple[int, int]:
    offset = dataset.timeline_offset
    ws = dataset.window_size
    win_start = tick_start - offset - (ws - 1)
    win_end = tick_end - offset - (ws - 1)
    return win_start, win_end


def _label_entropy(labels: np.ndarray) -> float:
    if len(labels) == 0:
        return 0.0
    counts = np.bincount(labels, minlength=3).astype(np.float64)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum())


def select_signal_trace_window(dataset: FI2010Dataset, cfg: dict) -> SignalTraceWindow | None:
    """Pick a contiguous validation window for the signal-trace plot."""
    length = cfg.get("window_length", 150)
    mode = cfg.get("mode", "auto")
    stock_index = cfg.get("stock", 2)

    if len(dataset) < length:
        return None

    if mode == "fixed":
        tick_start = cfg["tick_start"]
        tick_end = cfg["tick_end"]
        win_start, win_end = _ticks_to_window(dataset, tick_start, tick_end)
    else:
        best_score = -1.0
        win_start, win_end = 0, length - 1
        targets = dataset.windowed_targets()
        for start in range(0, len(dataset) - length + 1):
            end = start + length - 1
            hard = (
                dataset.volume_shock_mask[start:end + 1].sum()
                + dataset.depth_divergence_mask[start:end + 1].sum()
            )
            if hard < cfg.get("min_hard_cases", 1):
                continue
            entropy = _label_entropy(targets[start:end + 1])
            flat_ratio = (targets[start:end + 1] == 1).mean()
            if flat_ratio > cfg.get("max_flat_ratio", 0.95):
                continue
            score = 2.0 * hard + entropy
            if score > best_score:
                best_score = score
                win_start, win_end = start, end

        if best_score < 0:
            win_start, win_end = 0, length - 1

    if win_start < 0 or win_end >= len(dataset):
        return None

    tick_start, tick_end = _window_to_ticks(dataset, win_start, win_end)
    return SignalTraceWindow(stock_index, win_start, win_end, tick_start, tick_end)


@torch.no_grad()
def collect_trace_predictions(
    dataset: FI2010Dataset,
    window: SignalTraceWindow,
    teacher,
    student,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Run inference on the selected validation slice."""
    ws = dataset.window_size
    horizon = dataset.prediction_horizon_idx
    indices = range(window.win_start, window.win_end + 1)

    windows = np.stack([
        np.ascontiguousarray(dataset.features[i: i + ws]) for i in indices
    ], axis=0)
    x = torch.from_numpy(windows).to(device)

    ground_truth = np.array([
        dataset.labels[i + ws - 1, horizon] for i in indices
    ], dtype=np.int64)
    volume_shock = dataset.volume_shock_mask[window.win_start: window.win_end + 1]
    depth_divergence = dataset.depth_divergence_mask[window.win_start: window.win_end + 1]

    teacher.eval()
    student.eval()
    teacher_preds = teacher(x).max(1)[1].cpu().numpy()
    student_preds = student(x).max(1)[1].cpu().numpy()

    plot_ticks = np.arange(window.tick_start, window.tick_end + 1)
    return {
        "plot_ticks": plot_ticks,
        "ground_truth": ground_truth,
        "teacher_preds": teacher_preds,
        "student_preds": student_preds,
        "volume_shock": volume_shock,
        "depth_divergence": depth_divergence,
    }


def render_signal_trace_figure(trace: dict[str, np.ndarray], title: str) -> plt.Figure:
    """Render the bottom execution-decisions panel only."""
    fig, ax = plt.subplots(figsize=(15, 4))
    t = trace["plot_ticks"]
    gt = trace["ground_truth"]

    ax.plot(t, gt, label="GROUND TRUTH", color="black", lw=3.0)
    ax.plot(t, trace["teacher_preds"], label="TEACHER (DeepLOB)", color="green", linestyle="-.")
    ax.plot(t, trace["student_preds"], label="STUDENT (MLP)", color="red", linestyle="--")

    hard_marker_size = 24
    for i, tick in enumerate(t):
        if trace["volume_shock"][i]:
            ax.scatter(tick, gt[i], color="crimson", s=hard_marker_size, zorder=5, marker="o")
        if trace["depth_divergence"][i]:
            ax.scatter(tick, gt[i], color="darkorange", s=hard_marker_size, zorder=5, marker="X")

    handles, labels = ax.get_legend_handles_labels()
    handles.extend([
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="crimson", markersize=5),
        plt.Line2D([0], [0], marker="X", color="none", markerfacecolor="darkorange", markersize=5),
    ])
    labels.extend(["Volume shock (hard)", "Depth divergence (hard)"])

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Up", "Flat", "Down"])
    ax.invert_yaxis()
    ax.set_ylabel("Execution Decisions")
    ax.set_xlabel("Stock-Relative Tick")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0, ncol=1)
    fig.subplots_adjust(right=0.82)
    return fig


def log_signal_trace(
    val_loader_dataset,
    teacher,
    student,
    device: torch.device,
    cfg: dict,
    epoch: int,
    key_prefix: str = "student_val",
) -> dict | None:
    """Build the signal-trace plot and log it to the active W&B run."""
    if not cfg.get("enabled", True):
        return None

    import wandb

    stock_index = cfg.get("stock", 2)
    try:
        dataset = get_val_dataset_for_stock(val_loader_dataset, stock_index)
    except ValueError as exc:
        print(f"[signal_trace] Skipping: {exc}")
        return None

    window = select_signal_trace_window(dataset, cfg)
    if window is None:
        print("[signal_trace] Skipping: could not select a valid window.")
        return None

    trace = collect_trace_predictions(dataset, window, teacher, student, device)
    title = (
        f"Signal Trace — stock {window.stock_index} "
        f"[ticks {window.tick_start}–{window.tick_end}] — epoch {epoch}"
    )
    fig = render_signal_trace_figure(trace, title)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    from PIL import Image
    wandb_image = wandb.Image(Image.open(buf), caption=title)
    student_agree = float((trace["student_preds"] == trace["teacher_preds"]).mean())
    student_acc = float((trace["student_preds"] == trace["ground_truth"]).mean())
    teacher_acc = float((trace["teacher_preds"] == trace["ground_truth"]).mean())

    meta = {
        "stock_index": window.stock_index,
        "tick_start": window.tick_start,
        "tick_end": window.tick_end,
        "mode": cfg.get("mode", "auto"),
    }
    wandb.log({
        f"{key_prefix}/signal_trace": wandb_image,
        f"{key_prefix}/signal_trace_student_teacher_agreement": student_agree,
        f"{key_prefix}/signal_trace_student_acc": student_acc,
        f"{key_prefix}/signal_trace_teacher_acc": teacher_acc,
        "epoch": epoch,
    })
    return meta
