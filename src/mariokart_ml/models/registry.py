import torch.nn as nn
from .conv import MarioKartCNN
from .lstm import MarioKartLSTM
from .model_impl import Model
from typing import Callable, Concatenate, ParamSpec, TypeVar, TypedDict, Optional
import torch
from pathlib import Path
from datetime import datetime
    
class CheckpointState(TypedDict):
    epoch: int
    state_dict: dict
    optimizer: dict
    scheduler: Optional[dict]
    model_name: str
    mdata: dict
    config: dict

MODEL_REGISTRY = {
    "cnn": MarioKartCNN,
    "lstm": MarioKartLSTM,
}

def is_registered(name: str):
    return name in MODEL_REGISTRY
    
R = TypeVar('R')
P = ParamSpec('P')
def registered_only(func: Callable[Concatenate[str, P], R]):
    def wrapper(name: str, *args: P.args, **kwargs: P.kwargs) -> R:
        if not is_registered(name):
            raise ValueError(f"Model '{name}' not found. Available: {list(MODEL_REGISTRY.keys())}")
        
        return func(name, *args, **kwargs)
        
    return wrapper


def register(model_type: type[Model], name: str):
    MODEL_REGISTRY[name] = model_type

@registered_only
def get(name, **kwargs) -> Model:
    return MODEL_REGISTRY[name](**kwargs)
        
@registered_only
def save(name: str, folder: Path, state: CheckpointState):
    assert state['model_name'] == name
    
    now = datetime.now()
    date_str = now.strftime("%Y_%m_%d")
    base_file_name = folder / f'{name}_checkpoint_{date_str}.ckpt'
    
    folder.mkdir(parents=True, exist_ok=True)
    files = list(folder.iterdir())
    files = list(filter(lambda x: str(base_file_name) in str(x), files))
    
    out_file_name, ext = tuple(str(base_file_name).split('.'))
    if len(files) > 0:
        out_file_name += f"_{str(len(files))}"
        
    out_file_name += '.' + ext
    
    torch.save(state, out_file_name)
    
@registered_only
def load(name: str, folder: Path, device: Optional[torch.device] = None):
    files = sorted(
        folder.iterdir(), 
        key=lambda f: f.stat().st_mtime, 
        reverse=True
    )
    checkpoint = torch.load(files[-1], map_location=(device or torch.device("cpu")))
    config = checkpoint['config']
    model = get(name, device=device, **config)
    model.load_state_dict(checkpoint['state_dict'])
    return model, checkpoint