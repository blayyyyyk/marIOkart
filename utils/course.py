from operator import pos
import torch
import os, sys
from mkds.utils import read_vector_2d, read_vector_3d, read_fx32, read_u16, read_u32
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.vector import get_mps_device, pairwise_distances_cross
from mkds.kcl import KCL

DEFAULT_NKM_PATH = "desert_course/src/course_map.nkm"


def get_section_header(data, addr=0x00):
    title = data[addr : addr + 0x04]
    title = title.decode("ascii")
    num_entries = read_u32(data, addr + 0x04)
    return title, num_entries, data[addr + 0x08 :]


def read_checkpoint_data(nkm_path: str = DEFAULT_NKM_PATH):
    header_offset = 0x4C
    with open(nkm_path, "rb") as file:
        content = file.read()
        checkpoint_offset = read_u32(content, 0x2C) + header_offset
        checkpoint_end = read_u32(content, 0x30) + header_offset
        checkpoint_data = content[checkpoint_offset:checkpoint_end]
        _, num_entries, checkpoint_data = get_section_header(checkpoint_data)

        entries = []
        for i in range(num_entries):
            start = i * 0x24
            end = start + 0x24
            entry_data = checkpoint_data[start:end]
            entry_dict = {
                "p0": read_vector_2d(entry_data, 0x00),
                "p1": read_vector_2d(entry_data, 0x08),
                "distance": read_fx32(entry_data, 0x18),
                "id": read_u16(entry_data, 0x20),
            }
            entries.append(entry_dict)

        return entries


DEFAULT_KCL_PATH = "desert_course/src/course_collision.kcl"


def read_collision_data(kcl_path: str = DEFAULT_KCL_PATH):
    with open(kcl_path, "rb") as f:
        content = f.read()
        pos_offset = read_u32(content, 0x00)
        section_pos_1 = read_u32(content, 0x04)
        section_pos_2 = read_u32(content, 0x08)
        section_pos_3 = read_u32(content, 0x0C)
        pos_end = min(section_pos_1, section_pos_2, section_pos_3)
        pos_data = content[pos_offset:pos_end]

        points = []
        for i in range(0, len(pos_data), 0x0C):
            x, y, z = read_vector_3d(pos_data, i)
            points.append([x, y, z])

        return points

def collision_data_tensor(data, device=None):
    return torch.tensor(data, device=device)

def checkpoint_data_tensor_2d(data, device=None):
    points_0 = []
    points_1 = []
    for entry in data:
        p0 = torch.tensor(entry["p0"], device=device)
        p1 = torch.tensor(entry["p1"], device=device)
        points_0.append(p0)
        points_1.append(p1)

    return torch.stack(points_0), torch.stack(points_1)

"""
Returns checkpoint points with y-positions interpolated using closest distance of collision points
"""
def checkpoint_data_tensor_3d(checkpoint, collision, device=None):
    N = checkpoint.shape[0]
    ones = torch.ones((N, 1), device=device)
    checkpoint = torch.cat([checkpoint[:, 0, None], ones, checkpoint[:, 1: None]], dim=-1)
    D = pairwise_distances_cross(checkpoint, collision)
    D_argmin = D.argmin(dim=1)
    y_vals = collision[D_argmin, 1]
    checkpoint[:, 1] = y_vals
    return checkpoint

def prepare_course_data(device=None):
    checkpoint = read_checkpoint_data()
    checkpoint_a, checkpoint_b = checkpoint_data_tensor_2d(checkpoint, device=device)
    checkpoint = torch.cat([checkpoint_a, checkpoint_b], dim=0)
    collision = read_collision_data()
    collision = collision_data_tensor(collision, device=device)
    checkpoint = checkpoint_data_tensor_3d(checkpoint, collision, device=device)
    checkpoint_a, checkpoint_b = checkpoint.chunk(2, dim=0)
    return checkpoint_a, checkpoint_b, collision


def read_kcl_triangles(path: str | None = None, device = None):
    kcl = None
    if path is not None:
        kcl = KCL.from_file(path)
    else:
        kcl = KCL.from_file()
        
    

    height = torch.tensor([k['height'] for k in kcl.prisms], dtype=torch.float32, device=device)
    pos_i = torch.tensor([k['pos_i'] for k in kcl.prisms], dtype=torch.int32, device=device)
    
    fnrm_i = torch.tensor([k['fnrm_i'] for k in kcl.prisms], dtype=torch.int32, device=device)
    enrm1_i = torch.tensor([k['enrm1_i'] for k in kcl.prisms], dtype=torch.int32, device=device)
    enrm2_i = torch.tensor([k['enrm2_i'] for k in kcl.prisms], dtype=torch.int32, device=device)
    enrm3_i = torch.tensor([k['enrm3_i'] for k in kcl.prisms], dtype=torch.int32, device=device)

    all_pos = torch.tensor(kcl._positions, dtype=torch.float32, device=device)
    all_norm = torch.tensor(kcl._normals, dtype=torch.float32, device=device)

    position = all_pos[pos_i]
    face_norm = all_norm[fnrm_i]
    edge_norm_0 = all_norm[enrm1_i]
    edge_norm_1 = all_norm[enrm2_i]
    edge_norm_2 = all_norm[enrm3_i]

    return decode_triangles(position, height, face_norm, edge_norm_0, edge_norm_1, edge_norm_2)
