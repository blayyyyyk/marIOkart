import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler, default_collate
import os
from pathlib import Path
import numpy as np
import json
from typing import cast
from torch.utils.data import DataLoader, random_split

class MarioKartDataset(Dataset):
    def __init__(self, folder_path: str, window: int):
        
        self.data = {}
        self.folder_path = Path(folder_path)
        self.mdata = {}
        self.window = window
        with open(self.folder_path / "mdata.json", "r") as f:
            self.mdata = json.load(f)
        
        self.batch_size = 0
        for key, value in self.mdata.items():
            path = self.folder_path / f"{key}.dat"
            assert os.path.exists(path), f"Data set file {path} is missing"
            self.data[key] = np.memmap(
                str(path), 
                dtype=value['dtype'], 
                mode='r', 
                shape=tuple(value['shape']) 
            )
            self.batch_size = value['shape'][0]
            if not ('std' in self.mdata[key] and 'mean' in self.mdata[key]): continue
            self.mdata[key]['std'] = torch.tensor(self.mdata[key]['std'])
            self.mdata[key]['mean'] = torch.tensor(self.mdata[key]['mean'])
            
    def __len__(self):
        return self.batch_size - self.window
        
    def __getitem__(self, idx):
        out = {}
        for key, arr in self.data.items():
            t = torch.from_numpy(arr[idx:idx+self.window])
            if 'std' in self.mdata[key] and 'mean' in self.mdata[key]:
                std = self.mdata[key]['std']
                mean = self.mdata[key]['mean']
                t = (t - mean) / std
                
            out[key] = t
        return out
        
class ConcatMarioKartDataset(ConcatDataset):
    def __init__(self, datasets: list[MarioKartDataset]):
        super(ConcatMarioKartDataset, self).__init__(datasets)
        self.mdata = self.compute_global_stats()
        
    @property
    def mk_datasets(self) -> list[MarioKartDataset]:
        """A properly typed alias for self.datasets"""
        return cast(list[MarioKartDataset], self.datasets)
        
    def compute_global_stats(self) -> dict:
        """
        Computes the pooled mean and standard deviation across all underlying datasets.
        Returns a dictionary mapping feature keys to their global 'mean' and 'std'.
        """
        if len(self.datasets) == 0:
            return {}

        global_stats = {}
        total_samples = sum(len(ds) for ds in self.mk_datasets)
        
        # We assume the first dataset has all the representative keys
        first_mdata = self.mk_datasets[0].mdata
        
        # Find which features actually have mean/std tracked
        keys_to_normalize = [
            k for k, v in first_mdata.items() 
            if 'mean' in v and 'std' in v
        ]

        for key in keys_to_normalize:
            # --- Pass 1: Compute Global Mean ---
            sum_weighted_means = 0.0
            
            for ds in self.mk_datasets:
                n = len(ds)
                # Ensure float32 for safe math
                mu = ds.mdata[key]['mean'].to(torch.float32)
                sum_weighted_means += n * mu
                
            global_mean = sum_weighted_means / total_samples
            
            # --- Pass 2: Compute Global Variance ---
            sum_weighted_vars = 0.0
            
            for ds in self.mk_datasets:
                n = len(ds)
                mu: torch.Tensor = ds.mdata[key]['mean'].to(torch.float32)
                std: torch.Tensor = ds.mdata[key]['std'].to(torch.float32)
                
                # Variance is std squared
                var = std ** 2 
                
                # Pooled variance formula
                sum_weighted_vars += n * (var + (mu - global_mean) ** 2)
                
            global_var: torch.Tensor = sum_weighted_vars / total_samples
            global_std = torch.sqrt(global_var)
            
            # Store the computed stats for this feature
            global_stats[key] = {
                'mean': global_mean,
                'std': global_std
            }
            
        return global_stats
            
def prepare_data(data_folders: list[str], batch_size, seq_len, split_ratio: float):
    # Concat all dataset sources
    datasets = []
    for path in data_folders:
        ds = MarioKartDataset(path, window=seq_len)
        datasets.append(ds)
    
    dataset = ConcatMarioKartDataset(datasets)
    
    # Make train/test split
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Random batch sequences
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    return train_dataloader, test_dataloader, dataset.mdata
    
if __name__ == "__main__":
    # Configuration
    BATCH_SIZE = 32
    SEQ_LEN = 16
    
    # 1. Init Dataset
    dataset_1 = MarioKartDataset("data/interim/f8c_pikalex", 32)
    dataset_2 = MarioKartDataset("data/interim/f8c_pikalex", 32)
    dataset = ConcatMarioKartDataset([dataset_1, dataset_2])
    print(len(dataset_1), len(dataset_2), len(dataset))
    
    # 3. Build the DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    # Test the output
    count = 0
    for batch in dataloader:
        # Expected output: torch.Size([32, 16, C, H, W])
        print(batch['wall_distances'].shape) 
        if count > 2: break
        count += 1