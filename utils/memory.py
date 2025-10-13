from desmume.emulator import SCREEN_WIDTH, DeSmuME
from mkds.kcl import read_fx32
from utils.kcl_torch import KCLTensor
from utils.nkm_torch import NKMTensor
from mkds.utils import read_u32, read_vector_3d_fx32, read_u16
import torch
import json
import math
from utils.vector import (
    pairwise_distances_cross,
    intersect_ray_line_2d,
    triangle_raycast_batch,
    sample_cone,
    triangle_altitude,
    project_to_screen as project,
    compute_model_view as model_view
)
from typing import Callable, Concatenate, TypeVar, ParamSpec

P = ParamSpec("P")
R = TypeVar("R")

SCREEN_WIDTH, SCREEN_HEIGHT = 256, 192
Z_FAR = 1000.0
Z_NEAR = 0.0
Z_SCALE = 10.0

RACER_PTR_ADDR = 0x0217ACF8
COURSE_ID_ADDR = 0x23CDCD8
OBJECTS_PTR_ADDR = 0x0217B588
CHECKPOINT_PTR_ADDR = 0x021755FC
CLOCK_DATA_PTR = 0x0217AA34

def frame_cache(
    func: Callable[Concatenate[DeSmuME, P], R],
) -> Callable[Concatenate[DeSmuME, P], R]:
    val = None
    frame_count = 0

    def wrapper(emu: DeSmuME, *args: P.args, **kwargs: P.kwargs) -> R:
        nonlocal frame_count, val
        if emu.get_ticks() != frame_count or val is None:
            frame_count = emu.get_ticks()
            val = func(emu, *args, **kwargs)

        return val

    return wrapper


def game_cache(
    func: Callable[Concatenate[DeSmuME, P], R],
) -> Callable[Concatenate[DeSmuME, P], R]:
    val = None

    def wrapper(emu: DeSmuME, *args: P.args, **kwargs: P.kwargs) -> R:
        nonlocal val
        if val is None:
            val = func(emu, *args, **kwargs)

        return val

    return wrapper

def z_clip_mask(x: torch.Tensor) -> torch.Tensor:
    return (x[:, 2] < -Z_NEAR) & (x[:, 2] > -Z_FAR)

@game_cache
def read_clock_ptr(emu: DeSmuME):
    return emu.memory.unsigned.read_long(CLOCK_DATA_PTR)

@frame_cache
def read_clock(emu: DeSmuME):
    addr = read_clock_ptr(emu)
    return emu.memory.signed.read_long(addr + 0x08) * 10

def get_current_course_id(emu: DeSmuME):
    return emu.memory.unsigned.read_byte(COURSE_ID_ADDR)


def get_course_path(id: int):
    """
    Get the path of the directory containing a course's data
    """
    course_id_lookup = None
    with open("utils/courses.json", "r") as f:
        course_id_lookup = json.load(f)

    assert course_id_lookup is not None
    assert str(id) in course_id_lookup
    return course_id_lookup[str(id)]


@game_cache
def load_current_kcl(emu: DeSmuME, device):
    assert device is not None
    id = get_current_course_id(emu)
    path = get_course_path(id)
    path = f"./courses/{path}/course_collision.kcl"
    kcl = KCLTensor.from_file(path, device=device)
    return kcl


@game_cache
def load_current_nkm(emu: DeSmuME, device):
    assert device is not None
    id = get_current_course_id(emu)
    path = get_course_path(id)
    path = f"./courses/{path}/course_map.nkm"
    nkm = NKMTensor.from_file(path, device=device)
    return nkm


def read_racer_ptr(emu: DeSmuME, addr: int = RACER_PTR_ADDR):
    return emu.memory.unsigned.read_long(addr)

@frame_cache
def read_position(emu: DeSmuME, device):
    data = emu.memory.unsigned
    addr = read_racer_ptr(emu)
    pos = read_vector_3d_fx32(data, addr + 0x80)
    return torch.tensor(pos, dtype=torch.float32, device=device)

@frame_cache
def read_direction(emu: DeSmuME, device):
    data = emu.memory.unsigned
    addr = read_racer_ptr(emu)
    pos = read_vector_3d_fx32(data, addr + 0x68)
    return torch.tensor(pos, dtype=torch.float32, device=device)





def read_objects_array_max_count(emu: DeSmuME, addr: int = OBJECTS_PTR_ADDR):
    return emu.memory.signed.read_long(addr + 0x08)


def read_objects_array_ptr(emu: DeSmuME, addr: int = OBJECTS_PTR_ADDR):
    return emu.memory.signed.read_long(addr + 0x10)


def read_object_offset(emu: DeSmuME, id: int):
    return read_objects_array_ptr(emu) + id * 0x1C


def read_object_ptr(emu: DeSmuME, id: int):
    offset = read_object_offset(emu, id)
    return emu.memory.unsigned.read_long(offset + 0x18)


def read_object_flags(emu: DeSmuME, id: int):
    offset = read_object_offset(emu, id)
    return emu.memory.unsigned.read_short(offset + 0x14)


def read_object_position_ptr(emu: DeSmuME, id: int):
    offset = read_object_offset(emu, id)
    return emu.memory.unsigned.read_long(offset + 0x0C)


def read_object_is_ignored(emu: DeSmuME, id: int):
    obj_ptr = read_object_ptr(emu, id)
    flags = read_object_flags(emu, id)
    return obj_ptr == 0 or flags & 0x200


def read_object_is_deleted(emu: DeSmuME, id: int):
    pos_ptr = read_object_position_ptr(emu, id)
    return pos_ptr == 0


def safe_object(func):
    def wrapper(emu: DeSmuME, id: int, *args, **kwargs):
        if read_object_is_deleted(emu, id):
            return None

        return func(emu, id, *args, **kwargs)

    return wrapper


@frame_cache
@safe_object
def read_object_position(emu: DeSmuME, id: int, device):
    pos_ptr = read_object_position_ptr(emu, id)
    pos = read_vector_3d_fx32(emu.memory.unsigned, pos_ptr)
    return torch.tensor(pos, device=device)


@frame_cache
@safe_object
def read_map_object_type_id(emu: DeSmuME, id: int):
    obj_ptr = read_object_ptr(emu, id)
    return emu.memory.signed.read_short(obj_ptr)


@frame_cache
@safe_object
def read_map_object_is_coin_collected(emu: DeSmuME, id: int):
    obj_ptr = read_object_ptr(emu, id)
    return emu.memory.unsigned.read_short(obj_ptr + 0x02) & 0x01 != 0


@frame_cache
@safe_object
def read_racer_object_is_ghost(emu: DeSmuME, id: int):
    obj_ptr = read_object_ptr(emu, id)
    ghost_flag = emu.memory.unsigned.read_byte(obj_ptr + 0x7C)
    return ghost_flag & 0x04 != 0


# Object flags
FLAG_DYNAMIC = 0x1000
FLAG_MAPOBJ = 0x2000
FLAG_ITEM = 0x4000
FLAG_RACER = 0x8000


@frame_cache
def read_objects(emu: DeSmuME):
    obj_ids: dict[str, list[int]] = {
        "map_objects": [],
        "racer_objects": [],
        "item_objects": [],
        "dynamic_objects": [],
    }
    max_count = read_objects_array_max_count(emu)
    count = 0
    idx = 0
    while idx < 255 and count != max_count:
        if read_object_is_deleted(emu, idx):
            continue
        else:
            count += 1

        if read_object_is_ignored(emu, idx):
            continue

        flags = read_object_flags(emu, idx)
        if flags & FLAG_MAPOBJ != 0:
            obj_ids["map_objects"].append(idx)
        elif flags & FLAG_RACER != 0:
            obj_ids["racer_objects"].append(idx)
        elif flags & FLAG_ITEM != 0:
            obj_ids["item_objects"].append(idx)
        elif flags & FLAG_DYNAMIC == 0:
            obj_ids["dynamic_objects"].append(idx)

        idx += 1

    return obj_ids


CAMERA_PTR_ADDR = 0x0217AA4C


@frame_cache
def read_camera_ptr(emu: DeSmuME, addr: int = CAMERA_PTR_ADDR):
    return emu.memory.unsigned.read_long(addr)


@frame_cache
def read_camera_fov(emu: DeSmuME):
    addr = read_camera_ptr(emu)
    return emu.memory.unsigned.read_short(addr + 0x60) * (2 *
        math.pi / 0x10000
    )


@frame_cache
def read_camera_aspect(emu: DeSmuME):
    addr = read_camera_ptr(emu)
    return read_fx32(emu.memory.unsigned, addr + 0x6C)


@frame_cache
def read_camera_position(emu: DeSmuME, device):
    addr = read_camera_ptr(emu)
    pos = read_vector_3d_fx32(emu.memory.unsigned, addr + 0x24)
    elevation = read_fx32(emu.memory.unsigned, addr + 0x178)
    pos = (pos[0], pos[1] + elevation, pos[2])
    return torch.tensor(pos, device=device)



def read_camera_target_position(emu: DeSmuME, device):
    addr = read_camera_ptr(emu)
    pos = read_vector_3d_fx32(emu.memory.unsigned, addr + 0x18)
    return torch.tensor(pos, device=device)



def _compute_orthonormal_basis(
    forward_vector_3d: torch.Tensor, reference_vector_3d: torch.Tensor | None=None, device=None
):
    if reference_vector_3d is None:
        reference_vector_3d = torch.tensor(
            [0.0, 1.0, 0.0],
            dtype=forward_vector_3d.dtype,
            device=forward_vector_3d.device,
        )

    right_vector_3d = torch.cross(forward_vector_3d, reference_vector_3d, dim=0)
    right_vector_3d /= right_vector_3d.norm()

    up_vector_3d = torch.cross(right_vector_3d, forward_vector_3d, dim=0)
    up_vector_3d /= up_vector_3d.norm()

    basis = torch.stack(
        [
            right_vector_3d,
            up_vector_3d,
            forward_vector_3d,
        ],
        dim=0,
    )

    return basis


def _compute_model_view(camera_pos: torch.Tensor, camera_target_pos: torch.Tensor, device):
    forward = camera_target_pos - camera_pos
    forward /= torch.norm(forward, dim=-1)

    rot = _compute_orthonormal_basis(forward, device=device)

    pos_proj = rot @ camera_pos.unsqueeze(-2).transpose(-1, -2)

    model_view = torch.eye(4, dtype=rot.dtype, device=device)
    model_view[:3, :3] = rot
    model_view[:3, 3] = -pos_proj.squeeze(-1)
    
    return model_view


@frame_cache
def read_model_view(emu: DeSmuME, device):
    camera_pos = read_camera_position(emu, device=device)
    camera_target_pos = read_camera_target_position(emu, device=device)
    return _compute_model_view(camera_pos, camera_target_pos, device=device)


def _project_to_screen(world_points, model_view, fov, aspect, device=None):
    N = world_points.shape[0]

    # Homogenize points
    ones = torch.ones((N, 1), device=device)
    world_points = torch.cat([world_points, ones], dim=-1)
    cam_space = (model_view @ world_points.T).T

    # Perspective projection
    f = torch.tan(torch.tensor(fov, device=device) / 2)

    if cam_space.shape[0] == 0:
        return torch.empty((0, 2), device=device)

    fov_h = math.tan(fov)
    fov_w = math.tan(fov) * aspect

    projection_matrix = torch.zeros((4, 4), device=device)
    projection_matrix[0, 0] = 1 / fov_w
    projection_matrix[1, 1] = 1 / fov_h
    projection_matrix[2, 2] = (Z_FAR + Z_NEAR) / (Z_NEAR - Z_FAR)
    projection_matrix[2, 3] = -(2 * Z_FAR * Z_NEAR) / (Z_NEAR - Z_FAR)
    projection_matrix[3, 2] = 1

    clip_space = (projection_matrix @ cam_space.T).T

    ndc = clip_space[:, :3] / clip_space[:, 3, None]

    screen_x = (ndc[:, 0] + 1) / 2 * SCREEN_WIDTH
    screen_y = (1 - ndc[:, 1]) / 2 * SCREEN_HEIGHT
    screen_depth = clip_space[:, 2]
    screen_depth_norm = -Z_FAR / (-Z_FAR + Z_SCALE * clip_space[:, 2])
    return torch.stack([screen_x, screen_y, screen_depth, screen_depth_norm], dim=-1)


def project_to_screen(emu: DeSmuME, points: torch.Tensor, device):
    model_view = read_model_view(emu, device=device)
    fov = read_camera_fov(emu)
    aspect = read_camera_aspect(emu)
    return _project_to_screen(points, model_view, fov, aspect, device=device)


# CHECKPOINT INFO #

@game_cache
def read_checkpoint_ptr(emu: DeSmuME, addr: int = CHECKPOINT_PTR_ADDR):
    return emu.memory.unsigned.read_long(addr)

@frame_cache
def read_current_checkpoint(emu: DeSmuME):
    addr = read_checkpoint_ptr(emu)
    return emu.memory.unsigned.read_byte(addr + 0x46)

@frame_cache
def read_current_key_checkpoint(emu: DeSmuME):
    addr = read_checkpoint_ptr(emu)
    return emu.memory.signed.read_byte(addr + 0x48)

@frame_cache
def read_ghost_checkpoint(emu: DeSmuME):
    addr = read_checkpoint_ptr(emu)
    return emu.memory.signed.read_byte(addr + 0xD2)

@frame_cache
def read_ghost_key_checkpoint(emu: DeSmuME):
    addr = read_checkpoint_ptr(emu)
    return emu.memory.signed.read_byte(addr + 0xD4)

@frame_cache
def read_current_lap(emu: DeSmuME):
    addr = read_checkpoint_ptr(emu)
    return emu.memory.signed.read_byte(addr + 0x38)

@frame_cache
def read_next_checkpoint(emu: DeSmuME, checkpoint_count: int):
    current_checkpoint = read_current_checkpoint(emu)
    next_checkpoint = current_checkpoint + 1
    if next_checkpoint != checkpoint_count:
        return next_checkpoint
    else:
        return 0


def _convert_2d_checkpoints(P: torch.Tensor, source: torch.Tensor, dim=0):
    dim_mask = torch.range(0, source.shape[1] - 1, 1) != dim

    D = pairwise_distances_cross(P, source[:, dim_mask])
    min_idx = D.argmin(dim=1)

    result = torch.ones(P.shape[0], P.shape[1] + 1, device=P.device)
    result[:, dim_mask] = P
    result[:, dim] = source[min_idx, dim]

    return result


@game_cache
def read_checkpoint_positions(emu: DeSmuME, device):
    nkm = load_current_nkm(emu, device=device)
    kcl = load_current_kcl(emu, device=device)
    floor_mask = kcl.prisms.is_floor == 1
    floor_points = kcl.triangles[floor_mask]
    floor_points = floor_points.reshape(floor_points.shape[0] * 3, 3)
    return torch.stack(
        [
            _convert_2d_checkpoints(nkm._CPOI.position1, floor_points, dim=1),
            _convert_2d_checkpoints(nkm._CPOI.position2, floor_points, dim=1),
        ],
        dim=1,
    )

@frame_cache
def read_next_checkpoint_position(emu: DeSmuME, device):
    nkm = load_current_nkm(emu, device=device)
    checkpoints = read_checkpoint_positions(emu, device)
    checkpoint_count = nkm._CPOI.entry_count
    next_checkpoint = read_next_checkpoint(emu, checkpoint_count)
    return checkpoints[next_checkpoint]

@frame_cache
def read_current_checkpoint_position(emu: DeSmuME, device):
    checkpoints = read_checkpoint_positions(emu, device=device)
    current_checkpoint = read_current_checkpoint(emu)
    return checkpoints[current_checkpoint]

@frame_cache
def read_facing_point_checkpoint(emu: DeSmuME, direction: torch.Tensor, device):
    position = read_position(emu, device=device)
    checkpoint = read_next_checkpoint_position(emu, device)
    mask_xz = torch.tensor([0, 2], dtype=torch.int32, device=device)
    pos_xz = position[mask_xz]
    dir_xz = direction[mask_xz]
    pxz_1, pxz_2 = checkpoint[:, mask_xz].chunk(2, dim=0)
    pxz_1 = pxz_1.squeeze(0)
    pxz_2 = pxz_2.squeeze(0)
    intersect, _ = intersect_ray_line_2d(pos_xz, dir_xz, pxz_1, pxz_2)
    intersect = torch.tensor([intersect[0], position[1], intersect[1]], device=device)
    return intersect

@frame_cache
def read_forward_distance_checkpoint(emu, device):
    direction = read_direction(emu, device=device)
    position = read_position(emu, device=device)
    ray_point = read_facing_point_checkpoint(emu, direction, device=device)
    return torch.norm(ray_point - position)

@frame_cache
def read_left_distance_checkpoint(emu, device):
    direction = read_direction(emu, device=device)
    position = read_position(emu, device=device)
    up_basis = -torch.tensor([0, 1.0, 0], device=device, dtype=torch.float32)
    left_basis = torch.cross(direction, up_basis)
    ray_point = read_facing_point_checkpoint(emu, left_basis, device=device)
    return torch.norm(ray_point - position)

@frame_cache
def read_direction_to_checkpoint(emu: DeSmuME, device):
    f = read_forward_distance_checkpoint(emu, device=device)
    l = read_left_distance_checkpoint(emu, device=device)
    angle = torch.atan(f / l)
    return angle


# OBSTACLE INFO #

@frame_cache
def read_facing_point_obstacle(
    emu: DeSmuME,
    position: torch.Tensor | None = None,
    direction: torch.Tensor | None = None,
    device=None,
):
    assert device is not None
    kcl = load_current_kcl(emu, device=device)
    triangles = kcl.triangles
    wall_mask = kcl.prisms.is_wall == 1
    offroad_mask = (
        (kcl.prisms.collision_type == 5)
        | (kcl.prisms.collision_type == 3)
        | (kcl.prisms.collision_type == 2)
    )
    triangles = triangles[wall_mask | offroad_mask]
    B = triangles.shape[0]

    if B == 0:
        return None

    v1, v2, v3 = triangles.chunk(3, dim=1)
    v1 = v1.squeeze(1)
    v2 = v2.squeeze(1)
    v3 = v3.squeeze(1)

    if position is None:
        position = read_position(emu, device=device)

    if direction is None:
        direction = read_direction(emu, device=device)

    ray_dir = direction / torch.norm(direction, keepdim=True)
    angle = torch.tensor(torch.pi / 24, device=device)
    ray_samples = sample_cone(ray_dir, angle, 50)
    ray_dir = ray_dir.reshape(1, 3)
    ray_samples = torch.cat([ray_dir, ray_samples], dim=0)

    ray_origin = position
    ray_origin = ray_origin.unsqueeze(0)
    ray_origin = ray_origin.reshape(1, 3)
    ray_origin_samples = ray_origin.repeat(ray_samples.shape[0], 1)

    points = triangle_raycast_batch(ray_origin_samples, ray_samples, v1, v2, v3)
    N, M, C = points.shape
    points = points.reshape(N * M, C)
    if points.shape[0] == 0:
        return None

    dist = torch.cdist(points, ray_origin)
    min_id = torch.argmin(dist)
    current_point_min = points[min_id]
    return current_point_min

@frame_cache
def read_forward_distance_obstacle(emu: DeSmuME, device) -> torch.Tensor:
    position = read_position(emu, device=device)
    ray_point = read_facing_point_obstacle(emu, device=device)
    if ray_point is None:
        return torch.tensor([float("inf")], device=device)

    dist = torch.sqrt(torch.sum((position - ray_point) ** 2, dim=0))
    return dist

@frame_cache
def read_left_distance_obstacle(emu: DeSmuME, device) -> torch.Tensor:
    position = read_position(emu, device=device)
    direction = read_direction(emu, device=device)
    up_basis = -torch.tensor([0, 1.0, 0], device=device, dtype=torch.float32)
    left_basis = torch.cross(direction, up_basis)
    ray_point = read_facing_point_obstacle(emu, direction=left_basis, device=device)
    if ray_point is None:
        return torch.tensor([float("inf")], device=device)

    dist = torch.sqrt(torch.sum((position - ray_point) ** 2, dim=0))
    return dist

@frame_cache
def read_right_distance_obstacle(emu: DeSmuME, device) -> torch.Tensor:
    position = read_position(emu, device=device)
    direction = read_direction(emu, device=device)
    up_basis = torch.tensor([0, 1.0, 0], device=device, dtype=torch.float32)
    right_basis = torch.cross(direction, up_basis)
    ray_point = read_facing_point_obstacle(emu, direction=right_basis, device=device)
    if ray_point is None:
        return torch.tensor([float("inf")], device=device)

    dist = torch.sqrt(torch.sum((position - ray_point) ** 2, dim=0))
    return dist

@frame_cache
def read_checkpoint_distance_altitude(emu: DeSmuME, device) -> torch.Tensor:
    next_checkpoint = read_next_checkpoint_position(emu, device=device)
    p1, p2 = next_checkpoint.chunk(2, dim=0)

    position = read_position(emu, device=device)
    a = torch.norm(p1 - position)
    b = torch.norm(p2 - position)
    return triangle_altitude(a, b)
