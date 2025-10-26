from typing import Any, Protocol, Self
from abc import abstractmethod
from desmume.emulator import DeSmuME
from torch._prims_common import DeviceLikeType
from src.core.memory import *

class Metric:
    """Interface for any metric collector used during training."""

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state at the start of an episode."""

    @abstractmethod
    def update(self, emu: DeSmuME, device: DeviceLikeType | None = None) -> None:
        """Called each frame to record or accumulate values."""

    @abstractmethod
    def collect(self) -> dict[str, float]:
        """Return scalar summary(s) at the end of the episode."""
            
 
       
class DistanceMetric(Metric):
    def __init__(self):
        self.prev_id: int | None = None
        self.curr_id: int | None = None
        self.next_id: int | None = None
        self.dist: float = 0
        
    def reset(self) -> None:
        self.curr_id = None
        self.dist = 0
        
    def update(self, emu: DeSmuME, device: DeviceLikeType | None = None) -> None:
        current_checkpoint_id = read_current_checkpoint(emu)
        nkm = load_current_nkm(emu, device=device)
        checkpoint_count = nkm._CPOI.entry_count
        
        if self.curr_id is None:
            self.curr_id = current_checkpoint_id
            self.prev_id = read_previous_checkpoint(emu, checkpoint_count)
            self.next_id = read_next_checkpoint(emu, checkpoint_count)
            return
            
        if self.curr_id == current_checkpoint_id:
            return
        
        scale = 1
        midpoint_1 = read_current_checkpoint_position(emu, device=device).sum(dim=0) / 2
        midpoint_2 = None
        
        if current_checkpoint_id == self.next_id:
            midpoint_2 = read_next_checkpoint_position(emu, device=device).sum(dim=0) / 2
        elif current_checkpoint_id == self.prev_id:
            midpoint_2 = read_previous_checkpoint_position(emu, device=device).sum(dim=0) / 2
            scale = -1
            
        assert midpoint_1 is not None and midpoint_2 is not None, "Midpoints should be calculated"
        self.dist += scale * torch.norm(midpoint_1 - midpoint_2).item()
            
        self.curr_id = current_checkpoint_id
        self.prev_id = read_previous_checkpoint(emu, checkpoint_count)
        self.next_id = read_next_checkpoint(emu, checkpoint_count)
        
    def collect(self) -> dict[str, float]:
        return {
            'distance': self.dist
        }
        
class SpeedMetric(Metric):
    def __init__(self, distance_metric: DistanceMetric):
        self.distance_metric = distance_metric
        self.speed: float = 0.0
        self.start_time = 0
        
    def reset(self):
        self.start_time = 0
            
    def update(self, emu: DeSmuME, device: DeviceLikeType | None = None) -> None:
        if self.distance_metric.curr_id is None:
            self.start_time = read_clock(emu)
            return
        
        end_time = read_clock(emu)
        if end_time - self.start_time == 0:
            return
            
        self.speed = self.distance_metric.dist / (end_time - self.start_time)
        
    def collect(self) -> dict[str, float]:
        return {
            'speed': self.speed
        }
        

class OffroadMetric(Metric):
    def __init__(self):
        self.offroad_dist = 0.0
        
    def reset(self) -> None:
        self.prev_position = None
        self.offroad_dist = 0.0
        self.is_offroad = False
        
    def update(self, emu: DeSmuME, device: DeviceLikeType | None = None):
        position = read_position(emu, device=device)
        if self.prev_position is None:
            self.prev_position = position
            return
        
        attr_mask = lambda x: ((x == 3) | (x == 2) | (x == 5))
        current_is_offroad = read_touching_prism_type(emu, attr_mask, device=device)
        
        if current_is_offroad == self.is_offroad:
            return
           
        if current_is_offroad:
            self.prev_position = position
        elif not current_is_offroad:
            self.offroad_dist += torch.norm(position - self.prev_position).item()
        
        self.is_offroad = current_is_offroad
            
        
    def collect(self) -> dict[str, float]:
        return {
            'offroad_dist': self.offroad_dist
        }
        
      
def collect_all(metrics: list[Metric]):
    out: dict[str, float] = {}
    for metric in metrics:
        d = metric.collect()
        k, v = list(d.items())[0]
        out[k] = v
        
    return out
    
def reset_all(metrics: list[Metric]):
    for metric in metrics:
        metric.reset()
        
        
class FitnessScorer(Protocol):
    def __call__(self, metrics: dict[str, float]) -> float:
        """Return a scalar fitness score from metric summary dict."""
        ...
        
def default_fitness_scorer(metrics: dict[str, float]) -> float:
    """Return a scalar fitness score from metric summary dict."""
    return metrics['distance']