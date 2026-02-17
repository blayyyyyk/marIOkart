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

class RaceDataset(Dataset):
    def __init__(self, folder_path, seq_len=32, stride=1, dilation=1):
        with open(f"{folder_path}/metadata.json", 'r') as f:
            self.metadata: Metadata = json.load(f)
        
        self.obs_data = np.memmap(f"{folder_path}/samples.dat", dtype=np.float32, mode="r").reshape(-1, len(self.metadata['mean']))
        self.act_data = np.memmap(f"{folder_path}/targets.dat", dtype=np.int32, mode="r")

        self.seq_len = seq_len
        self.stride = stride
        self.dilation = dilation
        self.window_span = (seq_len - 1) * dilation + 1

        self.valid_indices = range(0, len(self.obs_data) - self.window_span + 1, stride)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        
        end_idx = start_idx + self.window_span
        obs_seq = torch.from_numpy(self.obs_data[start_idx : end_idx : self.dilation])

        if start_idx == 0:
            future_prev_acts = torch.from_numpy(self.act_data[self.dilation - 1 : end_idx - 1 : self.dilation])
            prev_act_seq = torch.cat([torch.tensor([0]), future_prev_acts])
        else:
            prev_act_seq = torch.from_numpy(self.act_data[start_idx - 1 : end_idx - 1 : self.dilation])

        last_frame_idx = end_idx - 1
        target = torch.tensor(self.act_data[last_frame_idx], dtype=torch.long)

        return obs_seq, prev_act_seq, target

def combine_dataset_stats(datasets: list[RaceDataset]):
    total_count = 0
    total_sum = None
    total_sq_sum = None

    for ds in datasets:
        n = ds.metadata['size']
        mu = np.array(ds.metadata['mean'], dtype=np.float32)
        sigma = np.array(ds.metadata['std'], dtype=np.float32)
        
        # initialize accumulators on first loop
        if total_sum is None:
            total_sum = np.zeros_like(mu)
            total_sq_sum = np.zeros_like(mu)
            
        # update sum and square sum
        current_sum = n * mu
        current_sq_sum = n * (sigma**2 + mu**2)
        
        # increment cumulative sums
        total_count += n
        total_sum += current_sum
        total_sq_sum += current_sq_sum

    assert total_sum is not None and total_sq_sum is not None and total_count != 0

    # compute global mean, variance, and std
    global_mean = total_sum / total_count
    global_variance = (total_sq_sum / total_count) - (global_mean ** 2)
    global_std = np.sqrt(np.maximum(global_variance, 0)) # clip variance to 0 to avoid error
    
    return torch.tensor(global_mean), torch.tensor(global_std), torch.tensor(total_count)

class ConcatRaceDataset(ConcatDataset):
    def __init__(self, datasets: list[RaceDataset]):
        super().__init__(datasets)
        self.metadata = combine([x.metadata for x in datasets])

# lstm model for mariokart ds
class MarioKartLSTM(nn.Module):
    def __init__(self, obs_dim=3, num_classes=9, hidden_dim=128, embed_dim=64, num_layers=2):
        super(MarioKartLSTM, self).__init__()
        self.obs_dim = obs_dim

        # action embeddings (encodes action semantics)
        self.action_embed = nn.Embedding(num_classes, embed_dim)
        self.obs_embed = nn.Linear(obs_dim, embed_dim)
        
        # LSTM layer
        input_size = embed_dim * 2
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, batch_first=True, dropout=0.2)

        # projection layer (embedding space to probability space)
        self.fc = nn.Linear(hidden_dim, num_classes)

        # untrainable params for feature scaling
        self.register_buffer("mu", torch.ones((obs_dim,), dtype=torch.float32),) # mean
        self.register_buffer("sigma", torch.ones((obs_dim,), dtype=torch.float32)) # std

    def forward(self, obs, prev_actions):
        # obs shape: (batch, seq_len, obs_dim)
        # prev_actions shape: (batch, seq_len)
        device = obs.device
        prev_ids = action_id(prev_actions, device=device)
        
        act_embeddings = self.action_embed(prev_ids) # (batch, seq_len, embed_dim)
        
        # feature scaling (only the distance components)
        assert isinstance(self.mu, torch.Tensor) and isinstance(self.sigma, torch.Tensor)
        normalized_obs = (obs - self.mu.to(device)) / (self.sigma.to(device) + 1e-8)

        obs_embeddings = self.obs_embed(normalized_obs)

        # we pass the lstm the history of observations (including current) + history of actions (excluding current)
        input = torch.cat([obs_embeddings, act_embeddings], dim=2)
        
        lstm_out, _ = self.lstm(input)
        
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