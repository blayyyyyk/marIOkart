import torch.nn as nn
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.model_impl import Model

class MarioKartCNN(Model):
    def __init__(self, num_features, embed_size, embed_count, out_channels=64, 
                 stride=1, kernel_size=3, dilation=1, n_layers=1, device=None, **kwargs):
        
        # Pass configuration args up to Model so get_config() tracks them
        super(MarioKartCNN, self).__init__(
            num_features=num_features, 
            embed_size=embed_size, 
            embed_count=embed_count, 
            device=device,
            out_channels=out_channels,
            stride=stride,
            kernel_size=kernel_size,
            dilation=dilation,
            n_layers=n_layers,
            **kwargs
        )
        
        assert n_layers > 0, "must have at least one conv layer"
        
        in_channels = self.num_features + self.embed_size
        self.conv_seq = nn.Sequential()
        
        # First convolutional layer
        self.conv_seq.append(nn.Conv1d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride,
            dilation=dilation
        ))
        
        # Subsequent convolutional layers
        for _ in range(n_layers - 1):
            self.conv_seq.append(nn.Conv1d(
                in_channels=out_channels, 
                out_channels=out_channels, 
                kernel_size=kernel_size, 
                stride=stride,
                dilation=dilation
            ))
        
        # Projection layer (embedding space to probability space)
        self.fc = nn.Linear(out_channels, self.embed_count)
        
    def forward(self, data: dict[str, torch.Tensor], targets=None):
        # 2. Extract and embed features
        # Shape: (batch, seq_len, num_features + embed_size)
        features = self.combine_features(data)
        
        # 3. Permute for PyTorch Conv1d
        # Conv1d expects (batch, channels, seq_len)
        features = features.permute(0, 2, 1)
        
        # 4. Pass through CNN
        conv_out = self.conv_seq(features)
        
        # 5. Permute back: (batch, channels, seq_len) -> (batch, seq_len, channels)
        conv_out = conv_out.permute(0, 2, 1)
        
        # 6. Extract last hidden state for prediction (matching LSTM setup)
        last_step_out = conv_out[:, -1, :] 
        
        # 7. Convert to probability space
        logits = self.fc(last_step_out)
            
        # 9. Compute Loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss
        