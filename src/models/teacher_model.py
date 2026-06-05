import torch
import torch.nn as nn

class TeacherDeepLOB(nn.Module):
    def __init__(self, num_classes: int = 3):
        """
        Simplified DeepLOB-style architecture acting as the high-capacity Teacher model.
        Processes spatial LOB depth levels via 2D Convolutions and temporal dynamics via LSTM.

        Note: this is a reduced variant of Zhang et al. (2019) - it uses a single
        temporal convolution instead of the original parallel Inception module.
        """
        super(TeacherDeepLOB, self).__init__()
        
        # Convolutional block 1: Spatial features extraction from the LOB rows
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 2), stride=(1, 2)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.01)
        )
        
        # Convolutional block 2: Further rows abstraction
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 2), stride=(1, 2)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.01)
        )
        
        # Convolutional block 3: Mapping across the remaining depth levels (10 levels total)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 10)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01)
        )
        
        # Convolutional block 4: Micro-temporal modeling inside the window context
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1), padding=(1, 0)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01)
        )
        
        # Recurrent layer processing aggregated temporal features
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True)
        
        # Final classification head matching the 3 target classes
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape context: (Batch, Window_Size, Features) -> (B, T, 40)
        # Add channel dimension for Conv2d processing -> (B, 1, T, 40)
        x = x.unsqueeze(1)
        
        x = self.conv1(x)  # Out: (B, 16, T, 20)
        x = self.conv2(x)  # Out: (B, 16, T, 10)
        x = self.conv3(x)  # Out: (B, 32, T, 1)
        # conv4 has kernel (4, 1) with padding (1, 0), so the temporal axis
        # shrinks from T to T-1 (e.g. 10 -> 9).
        x = self.conv4(x)  # Out: (B, 32, T-1, 1)

        # Reshape spatial context tensor maps for LSTM sequence compliance
        x = x.squeeze(3)       # Out: (B, 32, T-1)
        x = x.transpose(1, 2)  # Out: (B, T-1, 32)
        
        # Forward sequence execution through LSTM layers
        lstm_out, _ = self.lstm(x)
        
        # Take the hidden representation from the last historical step (t = -1)
        out_space = lstm_out[:, -1, :]
        
        return self.fc(out_space)