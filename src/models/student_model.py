import torch
import torch.nn as nn

class StudentMLP(nn.Module):
    def __init__(self, window_size: int = 10, num_features: int = 40, num_classes: int = 3):
        """
        Lightweight Student model for ultra-low latency inference in LOB price prediction.
        Flattens the temporal input sequence and passes it through a shallow MLP.
        """
        super(StudentMLP, self).__init__()
        
        # Calculate the total input dimension after flattening (T * 40)
        input_dim = window_size * num_features
        
        # Shallow, high-performance architecture
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape x: (Batch, Window_Size, Features) -> (B, T, 40)
        # Flatten the sequence dimensions to (Batch, T * 40)
        x = x.view(x.size(0), -1)
        
        # Output logits for the 3 target classes (Up, Flat, Down)
        return self.mlp(x)