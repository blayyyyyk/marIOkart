import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, ConcatDataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import json, os
from typing import Optional
from src.core.emulator import Metadata, combine
from src.utils.vector import get_mps_device
from src.models.model_impl import Model

# default model hyperparams
CONFIG = {
    'seq_len': 128,          # Look back 32 frames (approx 0.5s at 60fps)           # Left, Forward, Right sensors
    'num_classes': 9,       # The 9 distinct controller states
    'hidden_dim': 128,      # Size of LSTM memory
    'embed_dim': 64,        # Dimension to represent "Previous Action"
    'batch_size': 64,
    'lr': 0.001,
    'epochs': 15,
    'stride': 1,             # Overlap: stride 1 means 31 shared frames
    'dilation': 1
}

KEYMAP = [0, 33, 289, 1, 257, 321, 801, 273, 17]
valid_keys = torch.tensor(KEYMAP)

def action_id(x: torch.Tensor, device=None):
    global valid_keys
    valid_keys = valid_keys.to(device)
    matches = x.unsqueeze(-1) == valid_keys.view(1, 1, -1)
    ids = matches.float().argmax(dim=-1)
    return ids.squeeze(-1)

def get_target_indices(raw_targets_batch, device=None):
    global valid_keys
    valid_keys = valid_keys.to(device)
    matches = (raw_targets_batch.unsqueeze(-1) == valid_keys.unsqueeze(0))
    target_indices = matches.float().argmax(dim=-1)
    return target_indices.long()

# lstm model for mariokart ds
class MarioKartLSTM(Model):
    def __init__(self, num_features, embed_size, embed_count, hidden_dim=64, num_layers=2):
        super(MarioKartLSTM, self).__init__(num_features, embed_size, embed_count)
        self.hidden_dim = hidden_dim
        
        # LSTM layer
        input_size = self.num_features + self.embed_size
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, batch_first=True, dropout=0.2)

        # projection layer (embedding space to probability space)
        self.fc = nn.Linear(hidden_dim, self.embed_count)

    def forward(self, data: dict[str, torch.Tensor]):
        # obs shape: (batch, seq_len, obs_dim)
        # prev_actions shape: (batch, seq_len)
        features = super().forward(data)
        
        lstm_out, _ = self.lstm(features)
        
        # last hidden state for prediction
        last_step_out = lstm_out[:, -1, :] 
        
        # convert to probability space
        logits = self.fc(last_step_out)
        return logits


def train_model(dataset_paths: list[str]):
    device = get_mps_device()
    print(f"Training on {device}...")

    # Data Loading
    datasets: list[RaceDataset] = []
    for path in dataset_paths:
        ds = RaceDataset(
            path, 
            seq_len=CONFIG['seq_len'], 
            stride=CONFIG['stride'],
            dilation=CONFIG['dilation']
        )
        datasets.append(ds)

    assert len(datasets) != 0
    concat_ds: ConcatRaceDataset = ConcatRaceDataset(datasets)
    train_size = int(0.8 * len(concat_ds))
    test_size = len(concat_ds) - train_size
    train_ds, test_ds = random_split(concat_ds, [train_size, test_size])
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False)

    # Model Init
    model = MarioKartLSTM(
        len(datasets[0].metadata['mean']), 
        CONFIG['num_classes'], 
        CONFIG['hidden_dim'], 
        CONFIG['embed_dim'],
        
    ).to(device)
    model.mu = torch.tensor(concat_ds.metadata['mean'], dtype=torch.float32)
    model.sigma = torch.tensor(concat_ds.metadata['std'], dtype=torch.float32)
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.CrossEntropyLoss()

    train_losses, test_losses = [], []

    for epoch in range(CONFIG['epochs']):
        # training
        model.train()
        total_train_loss = 0
        
        for obs, prev_act, target in train_loader:
            obs, prev_act, target = obs.to(device), prev_act.to(device), target.to(device)

            optimizer.zero_grad()
            logits = model(obs, prev_act)
            
            target_ids = get_target_indices(target)
            loss = criterion(logits, target_ids)
            loss.backward()
            
            # grad clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            total_train_loss += loss.item()

        # validation
        model.eval()
        total_test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for obs, prev_act, target in test_loader:
                obs, prev_act, target = obs.to(device), prev_act.to(device), target.to(device)
                
                logits = model(obs, prev_act)

                target_ids = get_target_indices(target)
                loss = criterion(logits, target_ids)
                total_test_loss += loss.item()
                
                # calculate accuracy
                _, predicted = torch.max(logits.data, 1)
                total += target.size(0)
                correct += (predicted == target_ids).sum().item()

        avg_train = total_train_loss / len(train_loader)
        avg_test = total_test_loss / len(test_loader)
        accuracy = 100 * correct / total
        
        train_losses.append(avg_train)
        test_losses.append(avg_test)
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train:.4f} | Test Loss: {avg_test:.4f} | Acc: {accuracy:.2f}%")

    # Plotting
    plt.plot(train_losses, label='Train')
    plt.plot(test_losses, label='Test')
    plt.legend()
    plt.title("LSTM Learning Curve")
    plt.show()
    
    return model

# helper function for inference
def get_action_keymask(model, obs_window, prev_action_window, class_to_keymask_map):
    """
    class_to_keymask_map: Dictionary mapping int (0-8) -> int (DeSmuME keymask)
    """
    model.eval()
    with torch.no_grad():
        # add batch dimension
        obs = torch.tensor(obs_window).unsqueeze(0) 
        prev_act = torch.tensor(prev_action_window).unsqueeze(0)
        
        logits = model(obs, prev_act)
        
        # get the single most likely class index
        probs = torch.nn.functional.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        
        # convert to keymask
        keymask = class_to_keymask_map[predicted_class]
        
        return keymask, probs.max().item()

if __name__ == "__main__":
    TRAINING_ROOT_DIR = "private/training_data"
    data_dirs = [x[0] for x in os.walk(TRAINING_ROOT_DIR)][1:]
    trained_model = train_model(data_dirs)
    torch.save(trained_model.state_dict(), "./private/checkpoints/checkpoint_e01282026.pth")