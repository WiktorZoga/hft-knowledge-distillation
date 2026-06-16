import numpy as np

def check_volume_shock(features, labels, idx, price_std_thresh=0.05, vol_skew_thresh=1.8):
    """Filter 1: High volume imbalance on L1 while mid-price stays stagnant (Iceberg order)."""
    start = max(0, idx - 10)
    end = min(len(features), idx + 10)
    mid_prices = (features[start:end, 0] + features[start:end, 2]) / 2.0
    
    price_std = np.std(mid_prices)
    vol_delta = np.abs(features[idx, 3] - features[idx, 1])
    is_not_flat = labels[idx] != 1
    
    return (price_std < price_std_thresh) and (vol_delta > vol_skew_thresh) and is_not_flat

def check_depth_divergence(features, labels, idx):
    """Filter 2: Order Book Divergence (Level 1 pressure contradicts deeper layers)."""
    l1_imb = np.sign(features[idx, 3] - features[idx, 1])
    l5_imb = np.sign(features[idx, 19] - features[idx, 17]) # L5 Bid Vol - L5 Ask Vol
    is_not_flat = labels[idx] != 1
    
    return (l1_imb != l5_imb) and (l1_imb != 0) and (l5_imb != 0) and is_not_flat