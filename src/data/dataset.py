import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from torch.utils.data import Dataset

class FI2010Dataset(Dataset):
    def __init__(self, file_paths: list, window_size: int = 10, prediction_horizon_idx: int = 0, 
                 mean: np.ndarray = None, std: np.ndarray = None, use_subset: bool = False):
        self.window_size = window_size
        self.prediction_horizon_idx = prediction_horizon_idx
        
        all_features = []
        all_labels = []
        
        for path in file_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Data file missing: {path}")
                
            data = np.loadtxt(path)
            if use_subset:
                data = data[:, :5000]
                
            data = data.T
            features = data[:, :40]
            labels = data[:, -5:] - 1
            
            all_features.append(features)
            all_labels.append(labels)
            
        self.features = np.vstack(all_features)
        self.labels = np.vstack(all_labels).astype(np.int64)
        
        if mean is None or std is None:
            self.mean = np.mean(self.features, axis=0)
            self.std = np.std(self.features, axis=0)
            self.std[self.std == 0] = 1.0
        else:
            self.mean = mean
            self.std = std
            
        self.features = (self.features - self.mean) / self.std
        self.num_samples = len(self.features) - self.window_size + 1

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        x = self.features[idx : idx + self.window_size]
        target_idx = idx + self.window_size - 1
        y = self.labels[target_idx, self.prediction_horizon_idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)