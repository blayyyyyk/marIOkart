from mkds.utils import read_fx16, read_u16, read_u32, read_fx32, read_vector_3d
from mkds.utils import read_s32

import math


RACER_DATA_PTR = 0x0217ACF8
CAMERA_DATA_PTR = 0x0217AA4C

def get_racer_pos(data):
    data_addr = read_u32(data, RACER_DATA_PTR)
    x, y, z = read_vector_3d(data, data_addr + 0x80)
    return x, y, z
    
def get_racer_dir(data):
    racer_data_addr = read_u32(data, RACER_DATA_PTR)
    x, y, z = read_vector_3d(data, racer_data_addr + 0x68)
    return x, y, z
    
def get_camera_fov(data, data_addr=None):
    if not data_addr:
        data_addr = read_u32(data, CAMERA_DATA_PTR)
        
    fov = read_u16(data, data_addr + 0x60) * (2 * math.pi / 0x10000)
    aspect = read_fx32(data, data_addr + 0x6C)
    
    return fov, aspect
    
def get_camera_position(data, data_addr=None):
    if not data_addr:
        data_addr = read_u32(data, CAMERA_DATA_PTR)
        
    x, y, z = read_vector_3d(data, data_addr + 0x24)
    elevation = read_fx32(data, data_addr + 0x178)
    z += elevation
    
    return x, y, z
    
def get_camera_target_position(data, data_addr=None):
    if not data_addr:
        data_addr = read_u32(data, CAMERA_DATA_PTR)
        
    x, y, z = read_vector_3d(data, data_addr + 0x18)
    
    return x, y, z