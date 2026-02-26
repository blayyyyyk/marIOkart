import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num_features: int, embed_size: int, embed_count):
        super(Model, self).__init__()
        self.num_features = num_features
        self.embed_size = embed_size
        self.embed_count = embed_count
        self.key_embed = nn.Embedding(embed_count, embed_size)
        
    def forward(self, data: dict[str, torch.Tensor]):
        distances, keymask = data["wall_distances"], data["keymask"].squeeze(2)
        keymask_embed = self.key_embed(keymask.to(torch.int)) # -> (B, T, embedding_size)
        features = torch.cat([distances, keymask_embed], dim=-1) # -> (B, T, obs_dim+embedding_size-1)
        return features