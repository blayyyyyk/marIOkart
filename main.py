import torch, os
from desmume.emulator import DeSmuME
from utils.emulator import init_desmume_with_overlay, draw_triangles, draw_points
from utils.vector import get_mps_device, sample_cone, triangle_raycast_batch, smooth_mean, interpolate, clipped_mean
from utils.racer import Racer
import math
import json
from mkds.utils import read_u8

device = get_mps_device()

course_parent_directory = "./courses"

def init_racer(emu: DeSmuME):
    global racer, current_point
    
    
    track_id = emu.memory.unsigned.read_byte(0x23cdcd8)
    course_id_lookup = None
    with open("utils/courses.json", 'r') as f:
        course_id_lookup = json.load(f)
        
    assert course_id_lookup is not None
    course_directory = f"{course_parent_directory}/{course_id_lookup[str(track_id)]}"
    kcl_path = f"{course_directory}/course_collision.kcl"
    nkm_path = f"{course_directory}/course_map.nkm"
    
    racer = Racer.from_path(
        kcl_path, 
        nkm_path, 
        z_near=0.0,
        z_far=2000.0,
        device=device,
    )
    
    current_point = torch.tensor([0, 0, 0], device=racer.device)

""" Display Collision Triangles in Overlay """
def collision_overlay(emu: DeSmuME):
    global racer
    
    
    racer.memory = emu.memory.unsigned
    indices = racer.kcl.search_triangles(racer.position)
    if indices is None or len(indices) == 0:
        return
    
    indices = torch.tensor(indices, dtype=torch.int32, device=device)
    triangles = racer.kcl.triangles
    color_map = [
        (racer.kcl.prisms.is_wall, lambda x: x == 1, (1, 0, 1)),
        #(racer.kcl.prisms.is_floor, lambda x: x == 1, (0, 0.5, 1)),
        (racer.kcl.prisms.collision_type, lambda x: ((x == 3) | (x == 2) | (x == 5)), (1, 0, 0.3)),
        
        
    ]
    
    for attr, cond, color in color_map:
        # filter triangles by attribute condition
        condition_mask = cond(attr[indices])
        indices_masked = indices[condition_mask]
        if indices_masked.shape[0] == 0:
            continue
        
        # project triangles to screen space
        v1, v2, v3 = triangles[indices_masked].chunk(3, dim=1)
        v1 = racer.project_to_screen(v1.squeeze(1))
        v2 = racer.project_to_screen(v2.squeeze(1))
        v3 = racer.project_to_screen(v3.squeeze(1))
        
        # clip z
        near = racer.camera_z_near
        far = racer.camera_z_far
        valid = lambda x: (x[:, 2] < -near) & (x[:, 2] > -far)
        valid_mask = valid(v1) & valid(v2) & valid(v3)
        v1 = v1[valid_mask].tolist()
        v2 = v2[valid_mask].tolist()
        v3 = v3[valid_mask].tolist()
        
        draw_triangles(v1, v2, v3, color=color)

""" Display Kart Raycasting """
def raycasting_overlay(emu):
    global racer, current_point
    racer.memory = emu.memory.unsigned
    
    current_point_min = racer.get_facing_point(racer.position, racer.direction)
    if current_point_min is None:
        return
        
    current_point = interpolate(current_point, current_point_min, 0.1)
    
    forward_dist = torch.sqrt(torch.sum((current_point_min - racer.position)**2, dim=0))
    left_dist = racer.get_left_distance()
    right_dist = racer.get_right_distance()
    
    print(f"Forward Distance: {forward_dist}\nLeft Distance: {left_dist}\nRight Distance: {right_dist}")
    
    
    
def camera_overlay(emu: DeSmuME):
    global racer
    racer.memory = emu.memory.unsigned
    
    camera_target = racer.camera_target_position
    points = racer.project_to_screen(camera_target.unsqueeze(0))
    points = points.tolist()
    draw_points(points, color=(1, 0, 0))
    
    
def player_overlay(emu: DeSmuME):
    racer.memory = emu.memory.unsigned

def main(emu: DeSmuME):
    os.system("clear")
    raycasting_overlay(emu)
    collision_overlay(emu)
    camera_overlay(emu)
        
    
    

if __name__ == "__main__":
    init_desmume_with_overlay("mariokart_ds.nds", main, init_racer)
    
