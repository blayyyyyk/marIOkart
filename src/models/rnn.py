import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from desmume.controls import keymask, Keys
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import json
import numpy as np


class MarioKartRNN(nn.Module):
    def __init__(self, input_size=3, hidden_size=32, output_size=11, num_layers=5, norm_scale=3000):
        super(MarioKartRNN, self).__init__()
        self.kwargs = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "output_size": output_size,
            "num_layers": num_layers,
            "norm_scale": norm_scale
        }
        
        # Built-in torch.nn.RNN module
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, nonlinearity="relu")
        # Final fully connected layer for discrete action space
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0=None):
        # x shape: (batch_size, sequence_length, input_size)
        out, h = self.rnn(x, h0)
        # Use the output from the final time step for classification
        out = self.fc(out[:, -1, :])
        return out, h

# --- Training Logic ---
def train_vanilla_rnn(dataset_path, batch_size=64, seq_len=32, epochs=250):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Initialize your custom RaceDataset
    dataset = RaceDataset(dataset_path, sample_dim=3, target_dim=11)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    print(dataset.length)

    # Initialize model
    model = MarioKartRNN(input_size=3, hidden_size=128, output_size=11).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.BCEWithLogitsLoss() # Standard for discrete action prediction

    print(f"Training Vanilla RNN on {device} for CHI PLAY Studies #1 & #2...")
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (samples, targets) in enumerate(loader):
            # Targets need to be long for CrossEntropyLoss and squeezed to (batch,)
            samples = samples.to(device).view(-1, seq_len, 3) 
            targets = targets.to(device).float()
            targets = targets[seq_len-1::seq_len]

            optimizer.zero_grad()
            output, _ = model(samples)
            
            # Loss measures how far prediction is from ground truth expert input
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(loader):.5f}")
    
    

    
def train_with_benchmark(dataset_path, batch_size=32, seq_len=32, epochs=25, color: str = "ff0000"):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    full_dataset = RaceDataset(dataset_path, sample_dim=3, target_dim=9)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_ds, test_ds = random_split(full_dataset, [train_size, test_size])
    max_dist = None
    with open(f"{dataset_path}/metadata.json") as f:
        max_dist = json.load(f)["upper_bound"]
    
    train_loader = DataLoader(train_ds, batch_size=batch_size * seq_len, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size * seq_len, shuffle=True, drop_last=True)

    model = MarioKartRNN(input_size=3, hidden_size=128, output_size=9, norm_scale=max_dist).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.BCEWithLogitsLoss()
    train_losses, test_losses = [], []

    for epoch in range(epochs):
        # Training Phase
        model.train()
        total_train_loss = 0
        for samples, targets in train_loader:
            samples = samples.to(device).view(-1, seq_len, 3)
            targets = targets.to(device).float()[seq_len-1::seq_len]
            norm_samples = samples / samples.norm(-1)
            optimizer.zero_grad()
            output, _ = model(norm_samples)
            loss = criterion(output, targets)
            loss.backward()
            
            optimizer.step()
            total_train_loss += loss.item()

        # Evaluation Phase (Unseen Data)
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for samples, targets in test_loader:
                samples = samples.to(device).view(-1, seq_len, 3)
                targets = targets.to(device).float()[seq_len-1::seq_len]
                norm_samples = samples / samples.norm(-1)
                output, _ = model(norm_samples)
                loss = criterion(output.softmax(-1), targets)
                total_test_loss += loss.item()

        avg_train = total_train_loss / len(train_loader)
        avg_test = total_test_loss / len(test_loader)
        train_losses.append(avg_train)
        test_losses.append(avg_test)

        print(f"Sequence: {seq_len} | Epoch {epoch+1} | Train Loss: {avg_train:.4f} | Test Loss: {avg_test:.4f}")

    # Plotting for your "Results" chapter
    line_train, = plt.plot(train_losses, label=f'Training Loss (Sequence Length={seq_len})')
    line_test, = plt.plot(test_losses, label=f'Test Loss (Sequence Length={seq_len})')
    line_train.set_color(f"#{color[:6]}ff")
    line_test.set_color(f"#{color[:6]}80")
    
    
    torch.save([model.kwargs, model.state_dict()], "./private/checkpoints/checkpoint_e01052026.pth")
    
    
def collect_sequence_chart(seqs: dict[int, str]):
    for k, v in seqs.items():
        train_with_benchmark("./private/training_data/f8c_pikalex", color=v, seq_len=k)
        
    plt.title('Learning Curve: Training vs. Test Loss')
    plt.legend()
    plt.show()
    

colors = {
    1: "c4d6b0",
    2: "477998",
    4: "291f1e",
    8: "f64740",
    16: "a3333d",
    32: "A333A3",
}

if __name__ == "__main__":
    train_with_benchmark("./private/training_data/f8c_pikalex",
        batch_size=4, 
        seq_len=128, 
        epochs=100
    )
    