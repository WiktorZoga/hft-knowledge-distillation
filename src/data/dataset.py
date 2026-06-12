"""FI2010 limit-order-book dataset.

Two on-disk layouts are supported:

* ``.csv`` - the pandas-exported FI2010 dump (``FI2010_train.csv`` /
  ``FI2010_test.csv``). Each file has a header row and a leading index column;
  after dropping the index 149 values remain per sample: 40 raw LOB features,
  104 derived features and 5 label columns (one per prediction horizon, values
  ``{1, 2, 3}``). The LOB features are already z-score normalised.
* ``.txt`` - the classic DeepLOB orientation ``(rows, samples)`` read with
  ``np.loadtxt`` and transposed.

Both are normalised to a per-sample matrix of shape ``(N, 149)`` before the
first 40 columns are taken as features and the last 5 as labels.
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

NUM_FEATURES = 40
NUM_LABELS = 5
NUM_STOCKS = 5


def split_by_stock(features: np.ndarray, labels: np.ndarray, n_stocks: int = NUM_STOCKS):
    """Split concatenated FI2010 arrays into per-stock segments.

    The Dst files concatenate 5 stocks vertically with no explicit marker, so
    the boundaries are recovered as the ``n_stocks - 1`` largest jumps in the
    L1 ask price (same heuristic as the lob-deep-learning reference repo).
    On both shipped CSVs the four largest jumps are ~5-200x bigger than the
    fifth, so the recovery is unambiguous.

    Returns:
        List of ``(features, labels)`` tuples, one per stock, in file order.
    """
    jumps = np.abs(np.diff(features[:, 0]))
    boundaries = np.sort(np.argsort(jumps)[-(n_stocks - 1):]) + 1
    edges = [0, *boundaries.tolist(), len(features)]
    return [(features[edges[i]: edges[i + 1]], labels[edges[i]: edges[i + 1]])
            for i in range(n_stocks)]


def load_fi2010_arrays(file_paths, use_subset: bool = False, subset_size: int = 5000):
    """Load one or more FI2010 files into per-sample feature/label arrays.

    Returns:
        features: ``float32`` array of shape ``(total_samples, 40)``.
        labels:   ``int64`` array of shape ``(total_samples, 5)``, values ``{0, 1, 2}``.
    """
    all_features, all_labels = [], []
    for path in file_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file missing: {path}")

        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            # Header + leading index column -> index_col=0 drops the index.
            data = pd.read_csv(path, index_col=0).to_numpy()
        else:
            data = np.loadtxt(path).T  # (rows, samples) -> (samples, rows)

        if use_subset:
            data = data[:subset_size, :]

        all_features.append(data[:, :NUM_FEATURES].astype(np.float32))
        # Labels are stored as {1, 2, 3}; shift to {0, 1, 2} for CrossEntropy.
        all_labels.append((data[:, -NUM_LABELS:] - 1).astype(np.int64))

    return np.vstack(all_features), np.vstack(all_labels)


class FI2010Dataset(Dataset):
    """Sliding-window dataset over pre-loaded FI2010 sample arrays.

    Normalisation statistics (``mean`` / ``std``) must be computed on the
    training split and passed explicitly to the validation/test splits to avoid
    information leakage.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray, window_size: int = 10,
                 prediction_horizon_idx: int = 0, mean: np.ndarray = None, std: np.ndarray = None):
        self.window_size = window_size
        self.prediction_horizon_idx = prediction_horizon_idx

        if mean is None or std is None:
            mean = features.mean(axis=0)
            std = features.std(axis=0)
            std[std == 0] = 1.0
        self.mean = mean
        self.std = std

        self.features = ((features - mean) / std).astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.num_samples = max(0, len(self.features) - window_size + 1)

    def __len__(self) -> int:
        return self.num_samples

    def windowed_targets(self) -> np.ndarray:
        """Labels actually consumed by ``__getitem__`` (horizon column, offset
        by ``window_size - 1``). Used for class-weight computation."""
        start = self.window_size - 1
        return self.labels[start: start + self.num_samples, self.prediction_horizon_idx]

    def __getitem__(self, idx: int):
        x = self.features[idx: idx + self.window_size]
        target_idx = idx + self.window_size - 1
        y = self.labels[target_idx, self.prediction_horizon_idx]
        return torch.from_numpy(np.ascontiguousarray(x)), torch.tensor(y, dtype=torch.long)
