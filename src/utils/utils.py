import random

import numpy as np
import torch


def log_confusion_matrix(y_true, y_pred, class_names, key_prefix: str):
    """Log a confusion matrix to the active W&B run.

    Logs both the interactive count panel and a row-normalised table (each row
    sums to 1, i.e. per-class recall on the diagonal). Returns the normalised
    matrix so callers can persist it alongside the run.
    """
    import wandb
    from sklearn.metrics import confusion_matrix

    labels = list(range(len(class_names)))
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    panel = wandb.plot.confusion_matrix(y_true=list(y_true), preds=list(y_pred),
                                        class_names=class_names)
    table = wandb.Table(
        columns=["true \\ pred", *class_names],
        data=[[name, *row] for name, row in zip(class_names, cm_norm.round(4).tolist())],
    )
    wandb.log({
        f"{key_prefix}/confusion_matrix": panel,
        f"{key_prefix}/confusion_matrix_normalized": table,
    })
    return cm_norm


def resolve_device(config_device: str = "cuda") -> torch.device:
    """Resolve the compute device, falling back gracefully when the requested
    accelerator is unavailable. Pass ``"cpu"`` to force CPU."""
    if config_device == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
