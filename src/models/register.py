import torch.nn as nn
from src.models import *

model_registry = {}

def register_model(model: type[nn.Module]):
    model_registry[model.__name__] = model