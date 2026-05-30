import torch

class QueueImbalanceBaseline:
    def __init__(self, threshold: float = 0.1):
        """
        Analytical baseline based on the Queue Imbalance index from Gould et al.
        Predicts price direction based on the top-level order book depth.
        """
        self.threshold = threshold

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Batch of LOB data with shape (Batch, Window_Size, 40)
        Returns:
            torch.Tensor: Predicted classes with shape (Batch,)
        """
        # In FI2010 features array, the first level of LOB is at the beginning.
        # Index 1 is Ask Volume, Index 3 is Bid Volume for the latest tick in the window.
        # x[:, -1, 1] -> Ask Volume of the most recent tick
        # x[:, -1, 3] -> Bid Volume of the most recent tick
        ask_vol = x[:, -1, 1]
        bid_vol = x[:, -1, 3]
        
        # Calculate Queue Imbalance index
        imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-8)
        
        # Initialize predictions array filled with class 1 (Flat / No Change)
        preds = torch.ones(x.size(0), dtype=torch.long, device=x.device)
        
        # Map imbalance values directly to target classes
        # Class 0: Up (Imbalance > threshold)
        # Class 2: Down (Imbalance < -threshold)
        preds[imbalance > self.threshold] = 0
        preds[imbalance < -self.threshold] = 2
        
        return preds