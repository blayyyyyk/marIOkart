from __future__ import annotations
from desmume.emulator import DeSmuME
from src.visualization.draw import draw_paragraph, draw_points, draw_text, draw_triangles, draw_lines
from src.utils.vector import interpolate
import torch
import numpy as np
from src.core.memory import *
from src.utils.vector import project_to_screen as _project_to_screen
import torch
from typing import TypeAlias, Union
from functools import wraps

DeviceLikeType: TypeAlias = Union[str, torch.device, int]

AVAILABLE_OVERLAYS: list[Callable[[DeSmuME, DeviceLikeType | None], None]] = []

def register_overlay(func: Callable[[DeSmuME, DeviceLikeType | None], None]):
    AVAILABLE_OVERLAYS.append(func)
    return func

""" Display Collision Triangles in Overlay """

@register_overlay
def collision_overlay(emu: DeSmuME, device: DeviceLikeType | None = None):

    kcl = load_current_kcl(emu, device=device)
    position = read_position(emu, device=device)
    indices = kcl.search_triangles(position)
    if indices is None or len(indices) == 0:
        return

    indices = torch.tensor(indices, dtype=torch.int32, device=device)
    triangles = kcl.triangles
    color_map = [
        (kcl.prisms.is_wall, lambda x: x == 1, (1, 0, 1)),
        # (racer.kcl.prisms.is_floor, lambda x: x == 1, (0, 0.5, 1)),
        (
            kcl.prisms.collision_type,
            lambda x: ((x == 3) | (x == 2) | (x == 5)),
            (1, 0, 0.3),
        ),
    ]

    for attr, cond, color in color_map:
        # filter triangles by attribute condition
        condition_mask = cond(attr[indices])
        indices_masked = indices[condition_mask]
        if indices_masked.shape[0] == 0:
            continue

        # project triangles to screen space
        v1, v2, v3 = triangles[indices_masked].chunk(3, dim=1)
        v1 = project_to_screen(emu, v1.squeeze(1), device=device)
        v2 = project_to_screen(emu, v2.squeeze(1), device=device)
        v3 = project_to_screen(emu, v3.squeeze(1), device=device)
        
        # clip z
        valid = lambda x: (x[:, 2] < -Z_NEAR) & (x[:, 2] > -Z_FAR)
        valid_mask = valid(v1) & valid(v2) & valid(v3)
        v1 = torch.cat([v1[:, :2], v1[:, 3, None]], dim=-1)
        v2 = torch.cat([v2[:, :2], v2[:, 3, None]], dim=-1)
        v3 = torch.cat([v3[:, :2], v3[:, 3, None]], dim=-1)
        v1_np = v1[valid_mask].detach().cpu().numpy()
        v2_np = v2[valid_mask].detach().cpu().numpy()
        v3_np = v3[valid_mask].detach().cpu().numpy()
        draw_triangles(v1_np, v2_np, v3_np, np.array(color))

    

""" Display Kart Raycasting """
current_point = None

@register_overlay
def raycasting_overlay(emu: DeSmuME, device: DeviceLikeType | None = None):
    global current_point
    if current_point is None:
        current_point = torch.tensor(
            [0.0, 0.0, 0.0], dtype=torch.float32, device=device
        )
        
    position = read_position(emu, device=device)

    current_point_min = read_facing_point_obstacle(emu, device=device)
    if current_point_min is None:
        return

    current_point = interpolate(current_point, current_point_min, 0.1)

    forward_dist = torch.sqrt(
        torch.sum((current_point_min - position) ** 2, dim=0)
    )
    left_dist = read_left_distance_obstacle(emu, device=device)
    right_dist = read_right_distance_obstacle(emu, device=device)

    # print(f"Forward Distance: {forward_dist}\nLeft Distance: {left_dist}\nRight Distance: {right_dist}")

@register_overlay
def camera_overlay(emu: DeSmuME, device: DeviceLikeType | None = None):
    global racer, current_point

    camera_target = read_camera_target_position(emu, device=device)
    points = project_to_screen(emu, camera_target.unsqueeze(0), device=device)
    points_np = points.detach().cpu().numpy()
    draw_points(points_np, colors=np.array([1.0, 0.0, 0.0]), radius_scale=5.0)


""" Displays an overlay of a line connecting checkpoint endpoints of the next checkpoint. """

@register_overlay
def checkpoint_overlay_1(emu: DeSmuME, device: DeviceLikeType | None = None):
    global current_point
    position = read_position(emu, device=device)
    
    checkpoint = read_next_checkpoint_position(emu, device=device)
    checkpoint[:, 1] = position[1]
    
    checkpoint_proj = project_to_screen(emu, checkpoint, device=device)
    # depth filter 1
    z_clip = z_clip_mask(checkpoint_proj)
    checkpoint_proj = checkpoint_proj[z_clip]
    if checkpoint_proj.shape[0] == 0:
        return
    elif checkpoint_proj.shape[0] == 1:
        p1_np = checkpoint_proj[None, 0, :3].detach().cpu().numpy()
        
        draw_points(p1_np, colors=np.array([0.0, 1.0, 0.0]), radius_scale=10.0)
        return

    # display depth norm, preserve depth in 3d
    depth_norm = checkpoint_proj[:, 3, None] / 3
    depth = checkpoint_proj[:, 2, None]
    checkpoint_proj = torch.cat([checkpoint_proj[:, :2], depth_norm, depth], dim=-1)
    p1_np, p2 = checkpoint_proj[:, :3].chunk(2, dim=0)
    p1_np = p1_np.detach().cpu().numpy()
    p2_np = p2.detach().cpu().numpy()
    draw_lines(p1_np, p2_np, colors=np.array([0.0, 1.0, 0.0]), stroke_width_scale=1.0)


""" Displays an overlay of a ray connecting the kart and the next checkpoint boundary. """

@register_overlay
def checkpoint_overlay_2(emu: DeSmuME, device: DeviceLikeType | None = None):
    position = read_position(emu, device=device)
    direction = read_direction(emu, device=device)
    intersect = read_facing_point_checkpoint(emu, direction, device=device)
    intersect = intersect.unsqueeze(0)
    intersect_proj = project_to_screen(emu, intersect, device=device)
    z_clip_mask_2 = z_clip_mask(intersect_proj)

    # depth filter 2
    intersect_proj = intersect_proj[z_clip_mask_2]
    if intersect_proj.shape[0] == 0:
        return

    # display depth norm, preserve depth in 3d
    depth_norm = intersect_proj[:, 3, None]
    depth = intersect_proj[:, 2, None]
    intersect_proj = torch.cat([intersect_proj[:, :2], depth_norm, depth], dim=-1)
    intersect_proj = intersect_proj[:, :3]
    intersect_proj_np = intersect_proj.detach().cpu().numpy()
    draw_points(intersect_proj_np, colors=np.array([0.0, 1.0, 0.0]), radius_scale=1.0)

    intersect_proj_np[0, 2] = 0.1
    pos_proj = project_to_screen(emu, position.unsqueeze(0), device=device)
    pos_proj = pos_proj[:, :3]
    pos_proj[:, 2] = 0.1
    pos_proj_np = pos_proj.detach().cpu().numpy()
    
    draw_lines(intersect_proj_np, pos_proj_np, colors=np.array([0.0, 0.0, 1.0]), stroke_width_scale=1.0)

@register_overlay
def player_overlay(emu: DeSmuME, device: DeviceLikeType | None = None):
    objects = read_objects(emu)

    objs = [[], [], [], []]

    colors = [(0.7, 0.1, 0.6), (0.1, 0.7, 0.6), (0.1, 0.6, 0.7), (0.6, 0.1, 0.7)]
    for i, (key, ids) in enumerate(objects.items()):
        positions = []
        for id in ids:
            positions.append(read_object_position(emu, id, device=device))

        if len(positions) == 0: continue
        positions = torch.stack(positions, dim=0)
        object_positions = project_to_screen(emu, positions, device=device)

        z_clip = z_clip_mask(object_positions)
        object_positions = object_positions[z_clip]
        if object_positions.shape[0] == 0:
            continue

        object_positions = torch.cat(
            [object_positions[:, :2], object_positions[:, 3, None]], dim=-1
        )

        object_positions_np = object_positions.detach().cpu().numpy()
        colors_np = np.array(colors[i])
        draw_points(object_positions_np, colors=colors_np, radius_scale=5.0)
    