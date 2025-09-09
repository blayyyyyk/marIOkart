import os, math, torch
import numpy as np
import torch.nn.functional as F

from utils.emulator import init_desmume
from utils.keyboard import init_keyhack
from utils.memory import (
    read_fx32,
    read_vector_2d,
    ADDR_ORIENTATION_X,
    ADDR_ORIENTATION_Z,
    ADDR_POSITION,
    read_vector_3d,
)
from utils.course import read_collision_data, read_checkpoint_data
from utils.vector import get_mps_device, pairwise_distances, get_rays_2d

device = get_mps_device()

emu, window = init_desmume("mariokart_ds.nds")
init_keyhack(emu)

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

def main():
    # Run the emulation as fast as possible until quit
    while not window.has_quit():
        window.process_input()  # Controls are the default DeSmuME controls, see below.
        emu.cycle()
        window.draw()

        os.system("clear")

        pos_x, _, pos_z = read_vector_3d(emu.memory.unsigned, ADDR_POSITION)
        pos_2d = torch.tensor([pos_x, pos_z], device=device)
        dir_x, dir_z = read_vector_2d(
            emu.memory.unsigned, ADDR_ORIENTATION_X, ADDR_ORIENTATION_Z
        )
        dir_2d = torch.tensor([dir_x, dir_z], device=device)

        t0, u0 = get_rays_2d(pos_2d, dir_2d, pts_2d_0)
        t1, u1 = get_rays_2d(pos_2d, dir_2d, pts_2d_1)
        t = torch.cat([t0, t1], dim=0)
        print(f"Distance To Obstacle: {t.min()}")


if __name__ == "__main__":
    main()
