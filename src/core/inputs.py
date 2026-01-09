from __future__ import annotations
from abc import abstractmethod
from typing import Any, Protocol, TypeVar, ParamSpec, Callable
import torch
from private.mkds import driver_t
from src.core.memory import read_VecFx32

P = ParamSpec('P')
R = TypeVar('R')


AVAILABLE_INPUTS: list[str] = []

def register_overlay(func: Callable[P, torch.Tensor]):
    AVAILABLE_INPUTS.append(func.__name__)
    return func

def signed_plane_distance(plane_p: torch.Tensor, plane_n: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    # plane_p: (3,)
    # plane_n: (3,)
    # points: (B, 3,)
    # returns: (B,)
    signed_dist = (points - plane_p) @ plane_n
    return signed_dist
    
def reduce_to_local_subspace(driver: driver_t, points: torch.Tensor, tau: float) -> torch.Tensor:
    device = points.device
    
    # Filter for course points that lie near (plane signed distance threshold = (-tau, tau)) the driver's local xz-plane
    pos = read_VecFx32(driver.position, device)
    up = read_VecFx32(driver.upDir, device)
    signed_dist = (points - pos) @ up # distance from kart's local xz-plane
    mask = signed_dist <= tau
    kept_points = points[mask]
    kept_dist = signed_dist[mask]
    
    # Get local xz-plane coordinates
    fwd = read_VecFx32(driver.forwardDir, device)
    right = torch.cross(up, fwd)
    xz_proj = kept_points - kept_dist[:, None] * up
    xz = xz_proj - pos
    reduced_xz = torch.stack([
        xz @ right,
        xz @ fwd
    ], dim=1)
    
    return reduced_xz
    
def estimate_lr_partition(l0: torch.Tensor, l1: torch.Tensor, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    v = l1 - l0
    cp = torch.cross(v, points - l0)
    mask_l = cp > 0
    mask_r = cp < 0
    
    return points[mask_l], points[mask_r]
    


    