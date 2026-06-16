"""Build train / validation / test dataloaders for the FI2010 dataset."""

import glob
import os
from pathlib import Path

import numpy as np
import yaml
from torch.utils.data import ConcatDataset, DataLoader

from src.data.dataset import FI2010Dataset, NUM_STOCKS, load_fi2010_arrays, split_by_stock


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def load_yaml_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_stock_indices(stock_cfg) -> list:
    """Normalise the `stock` config value to a list of stock indices 0..4."""
    if stock_cfg is None or stock_cfg == "all":
        return list(range(NUM_STOCKS))
    if isinstance(stock_cfg, int):
        indices = [stock_cfg]
    else:
        indices = [int(s) for s in stock_cfg]
    for idx in indices:
        if not 0 <= idx < NUM_STOCKS:
            raise ValueError(f"stock index {idx} outside valid range 0..{NUM_STOCKS - 1}")
    return indices


def create_dataloaders(overrides: dict = None):
    """Build the three dataloaders.

    Args:
        overrides: optional dataset-config overrides (e.g. ``{"stock": 2,
            "prediction_horizon_idx": 3}``) so sweep scripts can vary the
            stock/horizon without editing the YAML.

    The FI2010 train/test dumps each concatenate 5 stocks vertically, so the
    arrays are first split per stock and every split (train/val/test) is built
    from per-stock windowed datasets. This guarantees that (a) sliding windows
    never span two different stocks and (b) the temporal train/val split is
    applied within each stock instead of carving the validation set out of
    whichever stocks happen to sit at the end of the file.
    """
    root_dir = get_project_root()
    main_cfg = load_yaml_config(root_dir / "config" / "config.yaml")
    data_cfg = load_yaml_config(root_dir / "config" / "dataset" / "fi2010.yaml")
    if overrides:
        data_cfg.update(overrides)

    raw_dir = root_dir / data_cfg["raw_data_dir"]
    use_subset = data_cfg.get("use_subset", main_cfg["development"]["use_subset"])
    subset_size = data_cfg.get("subset_size", 5000)
    window_size = data_cfg["window_size"]
    horizon_idx = data_cfg["prediction_horizon_idx"]
    train_ratio = data_cfg["train_ratio"]
    batch_size = data_cfg["batch_size"]
    stock_indices = resolve_stock_indices(data_cfg.get("stock", "all"))

    # Prefer the pandas-exported CSVs; fall back to the classic Dst .txt files.
    train_files = sorted(glob.glob(os.path.join(raw_dir, "*train*.csv"))
                         or glob.glob(os.path.join(raw_dir, "*Train_Dst*.txt")))
    test_files = sorted(glob.glob(os.path.join(raw_dir, "*test*.csv"))
                        or glob.glob(os.path.join(raw_dir, "*Test_Dst*.txt")))

    if not train_files or not test_files:
        raise RuntimeError(
            f"Source files missing inside target directory: {raw_dir}. "
            f"Run `python scripts/download_data.py` to fetch the FI2010 CSVs."
        )

    # Stock splitting needs the full arrays (boundaries are global), so the dev
    # subset is applied per selected stock *after* the split.
    train_segments = split_by_stock(*load_fi2010_arrays(train_files))
    test_segments = split_by_stock(*load_fi2010_arrays(test_files))

    train_parts, val_parts, test_parts = [], [], []
    val_offsets = []
    for idx in stock_indices:
        feats, labels = train_segments[idx]
        if use_subset:
            feats, labels = feats[:subset_size], labels[:subset_size]
        split_idx = int(len(feats) * train_ratio)
        val_offsets.append(split_idx)
        train_parts.append((feats[:split_idx], labels[:split_idx]))
        val_parts.append((feats[split_idx:], labels[split_idx:]))

        test_feats, test_labels = test_segments[idx]
        if use_subset:
            test_feats, test_labels = test_feats[:subset_size], test_labels[:subset_size]
        test_parts.append((test_feats, test_labels))

    # Normalisation statistics from the pooled training portions only; passed
    # explicitly to every split to avoid information leakage.
    pooled_train = np.vstack([feats for feats, _ in train_parts])
    mean = pooled_train.mean(axis=0)
    std = pooled_train.std(axis=0)
    std[std == 0] = 1.0

    hardness_cfg = data_cfg.get("hardness")

    def build(parts, timeline_offsets, stock_idxs):
        datasets = [
            FI2010Dataset(
                feats, labels, window_size=window_size,
                prediction_horizon_idx=horizon_idx, mean=mean, std=std,
                hardness_cfg=hardness_cfg, timeline_offset=offset, stock_index=stock_idx,
            )
            for (feats, labels), offset, stock_idx in zip(parts, timeline_offsets, stock_idxs)
        ]
        return ConcatDataset(datasets)

    train_offsets = [0] * len(train_parts)

    train_loader = DataLoader(
        build(train_parts, train_offsets, stock_indices),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        build(val_parts, val_offsets, stock_indices),
        batch_size=batch_size, shuffle=False, drop_last=False,
    )
    test_loader = DataLoader(
        build(test_parts, [0] * len(test_parts), stock_indices),
        batch_size=batch_size, shuffle=False, drop_last=False,
    )

    return train_loader, val_loader, test_loader
