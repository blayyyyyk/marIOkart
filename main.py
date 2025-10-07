import torch, os
from desmume.emulator import SCREEN_HEIGHT, SCREEN_WIDTH, DeSmuME
from utils.emulator import init_desmume_with_overlay, draw_triangles, draw_points, Scene
from utils.object import DynamicObject, GameItemObject, MapObject, RacerObject
from utils.vector import get_mps_device, sample_cone, triangle_raycast_batch, smooth_mean, interpolate, clipped_mean, extrapolate, intersect_ray_line_2d
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
        z_far=1000.0,
        device=device,
    )
    
    current_point = torch.tensor([0, 0, 0], device=racer.device)

""" Display Collision Triangles in Overlay """
def collision_overlay(emu: DeSmuME, scene: Scene):
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
        v1 = torch.cat([v1[:, :2], v1[:, 3, None]], dim=-1)
        v2 = torch.cat([v2[:, :2], v2[:, 3, None]], dim=-1)
        v3 = torch.cat([v3[:, :2], v3[:, 3, None]], dim=-1)
        v1 = v1[valid_mask].tolist()
        v2 = v2[valid_mask].tolist()
        v3 = v3[valid_mask].tolist()
        scene.add_triangles(v1, v2, v3, color=color)

""" Display Kart Raycasting """
def raycasting_overlay(emu: DeSmuME, scene: Scene):
    global racer, current_point
    racer.memory = emu.memory.unsigned
    
    current_point_min = racer.get_facing_point_obstacle(racer.position, racer.direction)
    if current_point_min is None:
        return
        
    current_point = interpolate(current_point, current_point_min, 0.1)
    
    forward_dist = torch.sqrt(torch.sum((current_point_min - racer.position)**2, dim=0))
    left_dist = racer.get_left_distance_obstacle()
    right_dist = racer.get_right_distance_obstacle()
    
    #print(f"Forward Distance: {forward_dist}\nLeft Distance: {left_dist}\nRight Distance: {right_dist}")
    
    
    
def camera_overlay(emu: DeSmuME, scene: Scene):
    global racer
    racer.memory = emu.memory.unsigned
    
    
    camera_target = racer.camera_target_position
    points = racer.project_to_screen(camera_target.unsqueeze(0))
    points = points.tolist()
    draw_points(scene, points, color=(1, 0, 0))
    
    

    

""" Displays an overlay of a line connecting checkpoint endpoints of the next checkpoint. """
def checkpoint_overlay_1(emu: DeSmuME, scene: Scene):
    racer.memory = emu.memory.unsigned
    
    checkpoint = racer.get_next_checkpoint()
    checkpoint[:, 1] = racer.position[1]
    checkpoint_proj = racer.project_to_screen(checkpoint)
    
    # depth filter 1
    z_clip_mask = racer.z_clip_mask(checkpoint_proj)
    checkpoint_proj = checkpoint_proj[z_clip_mask]
    if checkpoint_proj.shape[0] == 0:
        return
    elif checkpoint_proj.shape[0] == 1:
        p1 = checkpoint_proj[0, :3].tolist()
        scene.add_points([p1], color=(0, 1.0, 0))
        return
        
    # display depth norm, preserve depth in 3d
    depth_norm = checkpoint_proj[:, 3, None] / 3
    depth = checkpoint_proj[:, 2, None]
    checkpoint_proj = torch.cat([checkpoint_proj[:, :2], depth_norm, depth], dim=-1)
    p1, p2 = checkpoint_proj[:, :3].chunk(2, dim=0)
    p1 = p1.tolist()
    p2 = p2.tolist()
    scene.add_lines(p1, p2, color=(0, 1.0, 0), stroke_width=None)
    
""" Displays an overlay of a ray connecting the kart and the next checkpoint boundary. """
def checkpoint_overlay_2(emu: DeSmuME, scene: Scene):
    racer.memory = emu.memory.unsigned
    
    intersect = racer.get_facing_point_checkpoint(racer.position, racer.direction)
    intersect = intersect.unsqueeze(0)
    intersect_proj = racer.project_to_screen(intersect)
    z_clip_mask_2 = racer.z_clip_mask(intersect_proj)
    
    # depth filter 2
    intersect_proj = intersect_proj[z_clip_mask_2]
    if intersect_proj.shape[0] == 0:
        return
    
    # display depth norm, preserve depth in 3d
    depth_norm = intersect_proj[:, 3, None]
    depth = intersect_proj[:, 2, None]
    intersect_proj = torch.cat([intersect_proj[:, :2], depth_norm, depth], dim=-1)
    intersect_proj = intersect_proj[:, :3].tolist()
    scene.add_points(intersect_proj, color=(0, 1.0, 0))
    
    intersect_proj[0][2] = 0.1
    pos_proj = racer.project_to_screen(racer.position.unsqueeze(0))
    pos_proj = pos_proj[:, :3]
    pos_proj[:, 2] = 0.1
    pos_proj = pos_proj.tolist()
    scene.add_lines(intersect_proj, pos_proj, color=(0, 0, 1.0))
    
    angle = racer.get_direction_checkpoint()
    print(f"Angle: {angle}")
    
    
def player_overlay(emu: DeSmuME, scene: Scene):
    racer.memory = emu.memory.unsigned
    
    objects = racer.get_objects_details()
    
    objs = [[], [], [], []]
    
    for key, obj in objects.items():
        if isinstance(obj, MapObject):
            objs[1].append(obj.position)
        elif isinstance(obj, RacerObject):
            objs[2].append(obj.position)
        elif isinstance(obj, GameItemObject):
            objs[3].append(obj.position)
        elif isinstance(obj, DynamicObject):
            objs[4].append(obj.position)
    
    
    objs_filter = []
    for i, v in enumerate(objs):
        if len(v) == 0:
            continue
            
        objs_filter.append(torch.tensor(v, device=racer.device))
        
    
    colors = [(0.7, 0.1, 0.6), (0.1, 0.7, 0.6), (0.1, 0.6, 0.7), (0.6, 0.1, 0.7)]
    
    for i, positions in enumerate(objs_filter):
        object_positions = racer.project_to_screen(positions)
        
        z_clip_mask = racer.z_clip_mask(object_positions)
        object_positions = object_positions[z_clip_mask]
        if object_positions.shape[0] == 0:
            continue
        
        object_positions = torch.cat([
            object_positions[:, :2],
            object_positions[:, 3, None]
        ], dim=-1)
        
        #print(object_positions)
        object_positions = object_positions.tolist()
        scene.add_points(object_positions, color=colors[i])
        
    

def main(emu: DeSmuME, scene: Scene):
    #os.system("clear")
    
    player_overlay(emu, scene)
    raycasting_overlay(emu, scene)
    collision_overlay(emu, scene)
    checkpoint_overlay_1(emu, scene)
    checkpoint_overlay_2(emu, scene)
        
    
    

if __name__ == "__main__":
    init_desmume_with_overlay("mariokart_ds.nds", main, init_racer)
    
