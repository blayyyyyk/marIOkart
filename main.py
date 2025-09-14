import os, math, torch
import numpy as np
import torch.nn.functional as F
import binascii
from tqdm import tqdm
import re
from utils.race_data import get_camera_target_position, get_camera_position, get_racer_pos, get_racer_dir, get_camera_fov
from utils.emulator import init_desmume_with_overlay, set_overlay_points, SCREEN_WIDTH, SCREEN_HEIGHT
from utils.memory import (
    read_vector_2d,
    ADDR_ORIENTATION_X,
    ADDR_ORIENTATION_Z,
    ADDR_POSITION,
    read_vector_3d,
    read_vector_4d
)
from utils.course import read_checkpoint_data
from utils.vector import compute_model_view, get_mps_device, project_to_screen, get_rays_2d, project_to_camera
from desmume.emulator import DeSmuME

import time

device = get_mps_device()


def collision_data_tensor(data, device=None):
    return torch.tensor(data, device=device)


def checkpoint_data_tensor(data, device=None):
    points_0 = []
    points_1 = []
    for entry in data:
        p0 = torch.tensor(entry["p0"], device=device)
        p1 = torch.tensor(entry["p1"], device=device)
        points_0.append(p0)
        points_1.append(p1)

    return torch.stack(points_0), torch.stack(points_1)


checkpoint_data = read_checkpoint_data()
pts_2d_0, pts_2d_1 = checkpoint_data_tensor(checkpoint_data, device=device)



start = time.time()
reset_threshold = 75



def main(emu: DeSmuME):
    global matches, start, num_captures
    # Run the emulation as fast as possible until quit
    #os.system("clear")

    data = emu.memory.unsigned
    pos_x, pos_y, pos_z = get_racer_pos(data)
    pos_2d = torch.tensor([pos_x, pos_z], device=device)
    dir_x, dir_y, dir_z = get_racer_dir(data)
    dir_2d = torch.tensor([dir_x, dir_z], device=device)

    t0, u0 = get_rays_2d(pos_2d, dir_2d, pts_2d_0)
    t1, u1 = get_rays_2d(pos_2d, dir_2d, pts_2d_1)
    t = torch.cat([t0, t1], dim=0).min()
    
    camera_pos = torch.tensor(get_camera_position(data), dtype=torch.float32, device=device)
    camera_target_pos = torch.tensor(get_camera_target_position(data), dtype=torch.float32, device=device)
    fov, aspect = get_camera_fov(data)
    height_offset = camera_pos[1] + 100
    
    
    
    N = pts_2d_0.shape[0]
    ones = torch.ones((N, 1), device=device)
    points_left = torch.cat([
        pts_2d_0[:, 0, None], 
        ones * height_offset, 
        pts_2d_0[:, 1, None]
    ], dim=-1)
    
    points_right = torch.cat([
        pts_2d_1[:, 0, None], 
        ones * height_offset, 
        pts_2d_1[:, 1, None]
    ], dim=-1)
    points_left = torch.cat([points_left, torch.tensor([pos_x, pos_y, pos_z], device=device)[None, :]], dim=0)
    points_right = torch.cat([points_right, torch.tensor([pos_x, pos_y, pos_z], device=device)[None, :]], dim=0)
    
    model_view = compute_model_view(camera_pos, camera_target_pos, device=device)
    projected_points_left = project_to_camera(points_left, model_view, fov, aspect, device=device)
    projected_points_right = project_to_camera(points_right, model_view, fov, aspect, device=device)
    
    
    #projected_points = torch.where(valid_bounds, projected_points, torch.zeros_like(projected_points, device=device))
    clamp_min = torch.tensor([0, 0], device=device)
    clamp_max = torch.tensor([SCREEN_WIDTH, SCREEN_HEIGHT], device=device)
    projected_points_left = torch.clamp(projected_points_left, min=clamp_min, max=clamp_max)
    projected_points_right = torch.clamp(projected_points_right, min=clamp_min, max=clamp_max)
    set_overlay_points(projected_points_left.tolist(), color="red")
    set_overlay_points(projected_points_right.tolist(), color="blue")
    

    """pos_3d = torch.tensor([pos_x, pos_y, pos_z], device=device)
    dir_3d = torch.tensor([dir_x, 0, dir_z], device=device)

    angle = math.atan2(dir_z, dir_x)

    ray_point = t * dir_3d
    ray_point = ray_point
    ray_point_4d = torch.cat([ray_point, torch.tensor([1], device=device)], dim=-1) # Homogenized point
    
    checkpoint_points = torch.cat([pts_2d_0[:, 0, None], torch.zeros((pts_2d_0.shape[0], 1,), device=device), pts_2d_0[:, 0, None]], dim=-1)
    checkpoint_points = pos_3d - checkpoint_points
    
    checkpoint_points = torch.cat([checkpoint_points, torch.ones((checkpoint_points.shape[0], 1,), device=device)], dim=-1)
    screen_point = project_to_screen(checkpoint_points, proj, SCREEN_WIDTH, SCREEN_HEIGHT)
    screen_point = screen_point.tolist()
    
    

    dir_4d = torch.tensor([dir_x, dir_y, dir_z], device=device)"""



    # 0x010B514EB8 - 0x01024F4000 = 0x09020EB8




if __name__ == "__main__":
    init_desmume_with_overlay("mariokart_ds.nds", main)
