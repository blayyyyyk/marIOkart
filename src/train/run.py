from torch.optim.optimizer import Optimizer
from src.models.model_impl import Model
from src.train.dataset import ConcatMarioKartDataset, RandomSequenceBatchSampler, SequenceCollate, MarioKartDataset
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch
from torch import optim
from typing import Optional, TypedDict, cast
from pathlib import Path
from src.models.registry import get_model



def prepare_data(data_folders: list[str], batch_size, seq_len, split_ratio: float):
    # Concat all dataset sources
    datasets = []
    for path in data_folders:
        ds = MarioKartDataset(path)
        datasets.append(ds)
    
    dataset = ConcatMarioKartDataset(datasets)
    
    # Make train/test split
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Random batch sequences
    batch_sampler_1 = RandomSequenceBatchSampler(len(train_dataset), batch_size, seq_len)
    collate_fn = SequenceCollate(batch_size, seq_len)
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler_1,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    batch_sampler_2 = RandomSequenceBatchSampler(len(test_dataset), batch_size, seq_len)
    collate_fn = SequenceCollate(batch_size, seq_len)
    test_dataloader = DataLoader(
        test_dataset,
        batch_sampler=batch_sampler_2,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    return train_dataloader, test_dataloader, dataset.mdata

def run(
    model: Model, 
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    num_epochs: int,
    optimizer: Optimizer,
    device: torch.device,
    report_interval: int = 1,
):
    train_losses, test_losses = [], []
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    
    for epoch in range(num_epochs):
        # training run
        model.train()
        total_train_loss = 0
        
        for data in train_dataloader:
            data = {k: v.to(device) for k, v in data.items()}
            optimizer.zero_grad()
            
            logits = model(data)
            target = data['keymask'].squeeze(2)[..., -1].to(torch.long)
            
            loss = criterion(logits, target)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            total_train_loss += loss.item()
            
        # eval run (for metrics)
        model.eval()
        total_test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_dataloader:
                data = {k: v.to(device) for k, v in data.items()}
                
                logits = model(data)
                target = data['keymask'].squeeze(2)[..., -1].to(torch.long)

                loss = criterion(logits, target)
                total_test_loss += loss.item()
                
                _, predicted = torch.max(logits.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
        if epoch % report_interval == 0:
            avg_train = total_train_loss / len(train_dataloader)
            avg_test = total_test_loss / len(test_dataloader)
            accuracy = 100 * correct / total
            
            train_losses.append(avg_train)
            test_losses.append(avg_test)
            print(f"Epoch {epoch+1} | Train Loss: {avg_train:.4f} | Test Loss: {avg_test:.4f} | Acc: {accuracy:.2f}%")
                
        
class CheckpointState(TypedDict):
    epoch: int
    state_dict: dict
    optimizer: dict
    scheduler: Optional[dict]
    model_name: str
    mdata: dict
    config: dict
        
def save_checkpoint(folder: Path, state: CheckpointState):
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(state, folder / 'checkpoint.ckpt')
    
def load_checkpoint(folder: Path, model_name: str, device: Optional[torch.device] = None)
    files = sorted(
        folder.iterdir(), 
        key=lambda f: f.stat().st_mtime, 
        reverse=True
    )
    checkpoint = torch.load(files[-1], map_location=(device or torch.device("cpu")))
    config = checkpoint['config']
    model = get_model(model_name, **config)
    model.load_state_dict(checkpoint['state_dict'])
    return model, checkpoint
    
    