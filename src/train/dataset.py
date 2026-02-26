import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler, default_collate
import os
from pathlib import Path
import numpy as np
import json
from typing import cast

class MarioKartDataset(Dataset):
    def __init__(self, folder_path: str):
        
        self.data = {}
        self.folder_path = Path(folder_path)
        self.mdata = {}
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
        return self.batch_size
        
    def __getitem__(self, idx):
        out = {}
        for key, arr in self.data.items():
            t = torch.from_numpy(arr[idx])
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
                mu = ds.mdata[key]['mean'].to(torch.float32)
                std = ds.mdata[key]['std'].to(torch.float32)
                
                # Variance is std squared
                var = std ** 2 
                
                # Pooled variance formula
                sum_weighted_vars += n * (var + (mu - global_mean) ** 2)
                
            global_var = sum_weighted_vars / total_samples
            global_std = torch.sqrt(global_var)
            
            # Store the computed stats for this feature
            global_stats[key] = {
                'mean': global_mean,
                'std': global_std
            }
            
        return global_stats

class RandomSequenceBatchSampler(Sampler):
    def __init__(self, dataset_length: int, batch_size: int, seq_len: int):
        self.dataset_length = dataset_length
        self.batch_size = batch_size
        self.seq_len = seq_len
        
        # The highest index we can start a sequence at without going out of bounds
        self.valid_starts = dataset_length - seq_len + 1
        self.num_batches = self.valid_starts // batch_size

    def __iter__(self):
        # Generate random start indices for the whole epoch
        starts = torch.randperm(self.valid_starts).tolist()

        # Group them into batches
        for i in range(self.num_batches):
            batch_starts = starts[i * self.batch_size : (i + 1) * self.batch_size]
            
            batch_indices = []
            for start in batch_starts:
                # Add the sequential chunk for each random start
                batch_indices.extend(range(start, start + self.seq_len))
                
            yield batch_indices

    def __len__(self):
        return self.num_batches
        
class SequenceCollate:
    def __init__(self, batch_size: int, seq_len: int):
        self.batch_size = batch_size
        self.seq_len = seq_len

    def __call__(self, batch_list):
        # 1. default_collate turns a list of dicts into a dict of batched tensors.
        # Shape will currently be: [batch_size * seq_len, Channels, Height, Width]
        collated = default_collate(batch_list)

        # 2. Reshape into [Batch, Seq_len, Channels, Height, Width]
        for key, tensor in collated.items():
            feature_dims = tensor.shape[1:] # Grab the spatial/feature dimensions
            collated[key] = tensor.view(self.batch_size, self.seq_len, *feature_dims)

        return collated
            
if __name__ == "__main__":
    # Configuration
    BATCH_SIZE = 32
    SEQ_LEN = 16
    
    # 1. Init Dataset
    dataset_1 = MarioKartDataset("data/interim/f8c_pikalex")
    dataset_2 = MarioKartDataset("data/interim/f8c_pikalex")
    dataset = ConcatRaceDataset([dataset_1, dataset_2])
    print(len(dataset_1), len(dataset_2), len(dataset))
    
    # 2. Init Sampler and Collater
    batch_sampler = RandomSequenceBatchSampler(len(dataset), BATCH_SIZE, SEQ_LEN)
    collate_fn = SequenceCollate(BATCH_SIZE, SEQ_LEN)
    
    # 3. Build the DataLoader
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Test the output
    count = 0
    for batch in dataloader:
        # Expected output: torch.Size([32, 16, C, H, W])
        print(batch['wall_distances']) 
        if count > 10: break
        count += 1