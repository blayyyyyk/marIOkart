import torch
import torch.nn as nn
from typing import Optional

class Model(nn.Module):
    def __init__(self, num_features: int, embed_size: int, embed_count: int, device: Optional[torch.device] = None, **kwargs):
        super(Model, self).__init__()
        self.num_features = num_features
        self.embed_size = embed_size
        self.embed_count = embed_count
        self.kwargs = kwargs
        self.key_embed = nn.Embedding(embed_count, embed_size, device=device)
        
    def combine_features(self, data: dict[str, torch.Tensor]):
        distances, keymask = data["wall_distances"], data["keymask"].squeeze(2)
        keymask = keymask.to(torch.int)
        keymask[keymask > self.embed_count] = 0
        keymask_embed = self.key_embed(keymask) # -> (B, T, embedding_size)
        features = torch.cat([distances, keymask_embed], dim=-1) # -> (B, T, obs_dim+embedding_size-1)
        return features
        
    def forward(self, data) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        ...
        
    def get_config(self):
        out = {
            "num_features": self.num_features,
            "embed_size": self.embed_size,
            "embed_count": self.embed_count,
        }
        for i, v in self.kwargs.items():
            if i == "device": continue
            out[i] = v
            
        return out
