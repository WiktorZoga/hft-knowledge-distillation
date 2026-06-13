"""Aggregate classification metrics on expert-defined hard LOB regimes."""

from __future__ import annotations

import torch
from sklearn.metrics import accuracy_score, f1_score


def count_hardness_samples(dataset) -> dict[str, int]:
    """Count hard windows across a FI2010Dataset or ConcatDataset."""
    sub_datasets = getattr(dataset, "datasets", [dataset])
    volume_shock = sum(int(d.volume_shock_mask.sum()) for d in sub_datasets)
    depth_divergence = sum(int(d.depth_divergence_mask.sum()) for d in sub_datasets)
    total = sum(len(d) for d in sub_datasets)
    return {
        "volume_shock_count": volume_shock,
        "depth_divergence_count": depth_divergence,
        "total_windows": total,
    }


def compute_hardness_metrics(
    targets: list[torch.Tensor],
    preds: list[torch.Tensor],
    hardness_flags: list[torch.Tensor],
    key_prefix: str,
) -> dict[str, float | int]:
    """Return W&B-ready metrics for volume-shock and depth-divergence subsets."""
    y_true = torch.cat(targets).numpy()
    y_pred = torch.cat(preds).numpy()
    flags = torch.cat(hardness_flags).numpy()

    payload: dict[str, float | int] = {}
    for name, mask in (
        ("volume_shock", flags[:, 0]),
        ("depth_divergence", flags[:, 1]),
    ):
        count = int(mask.sum())
        payload[f"{key_prefix}/hard/{name}_count"] = count
        if count == 0:
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        payload[f"{key_prefix}/hard/{name}_acc"] = accuracy_score(yt, yp)
        payload[f"{key_prefix}/hard/{name}_f1_macro"] = f1_score(
            yt, yp, average="macro", zero_division=0
        )

    return payload
