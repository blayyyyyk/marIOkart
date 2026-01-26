import torch.nn as nn

class MarioKartModel(nn.Module):
    def __init__(self, obs_dim: int, num_classes: int):
        super(nn.Module, self).__init__()
        self.obs_dim = obs_dim
        self.num_classes = num_classes


    