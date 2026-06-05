import random

import numpy as np
import torch


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
