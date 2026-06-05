"""Build train / validation / test dataloaders for the FI2010 dataset."""

import glob
import os
from pathlib import Path

import yaml
from torch.utils.data import DataLoader

from src.data.dataset import FI2010Dataset, load_fi2010_arrays


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def load_yaml_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_dataloaders():
    root_dir = get_project_root()
    main_cfg = load_yaml_config(root_dir / "config" / "config.yaml")
    data_cfg = load_yaml_config(root_dir / "config" / "dataset" / "fi2010.yaml")

    raw_dir = root_dir / data_cfg["raw_data_dir"]
    use_subset = main_cfg["development"]["use_subset"]
    window_size = data_cfg["window_size"]
    horizon_idx = data_cfg["prediction_horizon_idx"]
    train_ratio = data_cfg["train_ratio"]
    batch_size = data_cfg["batch_size"]

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

    # Load the full training pool, then split it *temporally*: validation samples
    # come strictly after training samples. This avoids look-ahead leakage and
    # guarantees the validation set is disjoint from training even when the
    # dataset ships as a single file (the previous code aliased val == train).
    train_features, train_labels = load_fi2010_arrays(train_files, use_subset)
    split_idx = int(len(train_features) * train_ratio)

    train_dataset = FI2010Dataset(
        train_features[:split_idx], train_labels[:split_idx],
        window_size=window_size, prediction_horizon_idx=horizon_idx,
    )
    val_dataset = FI2010Dataset(
        train_features[split_idx:], train_labels[split_idx:],
        window_size=window_size, prediction_horizon_idx=horizon_idx,
        mean=train_dataset.mean, std=train_dataset.std,
    )

    test_features, test_labels = load_fi2010_arrays(test_files, use_subset)
    test_dataset = FI2010Dataset(
        test_features, test_labels,
        window_size=window_size, prediction_horizon_idx=horizon_idx,
        mean=train_dataset.mean, std=train_dataset.std,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader
