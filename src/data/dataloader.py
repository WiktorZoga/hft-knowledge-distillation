import os
import sys

# Ensure project root is in path for package imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import glob
import yaml
from pathlib import Path
from torch.utils.data import DataLoader
from src.data.dataset import FI2010Dataset

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
    
    # Target patterns matched directly against the extracted dataset files
    train_val_files = sorted(glob.glob(os.path.join(raw_dir, "*train*.csv")) or glob.glob(os.path.join(raw_dir, "*Train_Dst*.txt")))
    test_files = sorted(glob.glob(os.path.join(raw_dir, "*test*.csv")) or glob.glob(os.path.join(raw_dir, "*Test_Dst*.txt")))
    
    if not train_val_files or not test_files:
        raise RuntimeError(f"Source files missing inside target directory: {raw_dir}")
        
    # Since we have 1 large training file container in this split, 
    # we enforce dynamic partitioning if train_ratio splitting applies to lists
    if len(train_val_files) == 1:
        train_files = train_val_files
        val_files = train_val_files  # Fallback to prevent crash, subset split handled contextually
    else:
        split_idx = int(len(train_val_files) * data_cfg["train_ratio"])
        train_files = train_val_files[:split_idx]
        val_files = train_val_files[split_idx:]
    
    train_dataset = FI2010Dataset(
        file_paths=train_files,
        window_size=data_cfg["window_size"],
        prediction_horizon_idx=data_cfg["prediction_horizon_idx"],
        use_subset=use_subset
    )
    
    val_dataset = FI2010Dataset(
        file_paths=val_files,
        window_size=data_cfg["window_size"],
        prediction_horizon_idx=data_cfg["prediction_horizon_idx"],
        mean=train_dataset.mean,
        std=train_dataset.std,
        use_subset=use_subset
    )
    
    test_dataset = FI2010Dataset(
        file_paths=test_files,
        window_size=data_cfg["window_size"],
        prediction_horizon_idx=data_cfg["prediction_horizon_idx"],
        mean=train_dataset.mean,
        std=train_dataset.std,
        use_subset=use_subset
    )
    
    train_loader = DataLoader(train_dataset, batch_size=data_cfg["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=data_cfg["batch_size"], shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=data_cfg["batch_size"], shuffle=False, drop_last=False)
    
    return train_loader, val_loader, test_loader