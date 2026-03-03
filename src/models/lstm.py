import torch
import torch.nn as nn
from src.models.model_impl import Model
import torch.nn.functional as F

# lstm model for mariokart ds
class MarioKartLSTM(Model):
    def __init__(self, num_features, embed_size, embed_count, hidden_dim=64, num_layers=2, device=None):
        super(MarioKartLSTM, self).__init__(num_features, embed_size, embed_count, device=device)
        self.hidden_dim = hidden_dim
        
        # LSTM layer
        input_size = self.num_features + self.embed_size
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, batch_first=True, dropout=0.2)

        # projection layer (embedding space to probability space)
        self.fc = nn.Linear(hidden_dim, self.embed_count)

    def forward(self, data: dict[str, torch.Tensor], targets=None):
        # obs shape: (batch, seq_len, obs_dim)
        # prev_actions shape: (batch, seq_len)
        features = self.combine_features(data)
        
        
        lstm_out, _ = self.lstm(features)
        
        # last hidden state for prediction
        last_step_out = lstm_out[:, -1, :] 
        
        # convert to probability space
        logits = self.fc(last_step_out)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss