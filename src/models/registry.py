import torch.nn as nn
from src.models.conv import MarioKartCNN
from src.models.lstm import MarioKartLSTM
from src.models.rnn import MarioKartRNN

MODEL_REGISTRY = {
    "cnn": MarioKartCNN,
    "lstm": MarioKartLSTM,
    "rnn": MarioKartRNN
}

def get_model(name, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)