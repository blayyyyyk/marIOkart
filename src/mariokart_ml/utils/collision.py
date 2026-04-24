from typing import Literal

import numpy as np
import trimesh
from desmume.emulator_mkds import MarioKart

from ..utils.vector import generate_plane_vectors, project_to_plane, raycast_2d

COLLISION_TYPES = {
    0: "Road",
    1: "Slippery Road",
    2: "Weak Offroad",
    3: "Offroad",
    4: "Sound Trigger",
    5: "Heavy Offroad",
    6: "Slippery Road 2",
    7: "Boost Panel",
    8: "Wall",
    9: "Invisible Wall (ignored by cameras)",
    10: "Out of Bounds",
    11: "Fall Boundary",
    12: "Jump Pad",
    13: "Road (no drivers)",
    14: "Wall (no drivers)",
    15: "Cannon Activator",
    16: "Edge Wall (does not collide when on the ground)",
    17: "Falls Water",
    18: "Boost Pad w/ Min Speed",
    19: "Loop Road",
    20: "Special Road",
    21: "Wall 3",
    22: "Force Recalculate Route",
}

ACCEPTED_ROADS = [
    0,  # road
    2,  # weak offroad
    1,  # slippery road
    3,  # offroad
    5,  # heavy offroad
    6,  # slippery road 2
]


def get_standing_triangle_id(emu: MarioKart) -> int:
    """
    Gets the triangle id of the course collision map (KCL), where the triangle is of closest proximity to the kart.

    Args:
        emu (MarioKart): The MarioKart emulator instance.
    Returns:
        int: The index of the closest triangle in the KCL.
    """
    position = emu.memory.driver_position
    triangles = emu.memory.kcl.triangular_faces  # (B, 3, 3)
    center = triangles.mean(axis=1)  # (B, 3)
    dist = np.linalg.norm(center - position, axis=-1)  # (B,)
    nearby_idx = np.argmin(dist).item()  # (1,)
    return nearby_idx


def make_compsite_road_mask(
    emu: MarioKart,
    mode: Literal["nearest"] | Literal["strict"] = "nearest",
    min_id: int = 0,
):
    col_data = emu.memory.collision_data
    col_type = col_data["prism_attribute"]["collision_type"]

    heirarchy = {ACCEPTED_ROADS[i]: ACCEPTED_ROADS[: i + 1] for i in range(len(ACCEPTED_ROADS))}
    if mode == "nearest":
        nearby_triangle_idx = get_standing_triangle_id(emu)
        nearby_col_type = col_type[nearby_triangle_idx]
        if nearby_col_type in heirarchy:
            accepted_road_types = heirarchy[nearby_col_type]
        else:
            accepted_road_types = ACCEPTED_ROADS
    elif mode == "strict":
        accepted_road_types = ACCEPTED_ROADS[0:1]
    else:
        raise ValueError(f"Invalid mode: {mode}")

    road_mask = np.isin(col_type, accepted_road_types)
    return road_mask


def get_road_lines(emu: MarioKart, road_mask: np.ndarray, precision: float = 0.05) -> np.ndarray:
    """
    Finds shared lines between road and non-road triangles.
    """

    triangles = emu.memory.kcl.triangular_faces
    triangles = np.round(triangles * precision) / precision

    # build mesh
    raw_vertices = triangles.reshape(-1, 3)
    raw_faces = np.arange(len(raw_vertices)).reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=raw_vertices, faces=raw_faces, process=True, maintain_order=True)

    # face adjacency lists
    f1 = mesh.face_adjacency[:, 0]
    f2 = mesh.face_adjacency[:, 1]

    # one face is road and neighboring face is off-road
    boundary_mask = road_mask[f1] != road_mask[f2]

    # vertex indices for those specific boundary edges
    boundary_edge_indices = mesh.face_adjacency_edges[boundary_mask]

    # map the indices back to 3D coordinates
    boundary_lines = mesh.vertices[boundary_edge_indices]  # (E, 2, 3)

    return boundary_lines


def compute_collision_dists(
    emu: MarioKart,
    mode: Literal["nearest"] | Literal["strict"] = "nearest",
    min_id: int = 0,
    min_dist: float = 0.0,
    max_dist: float = 3000.0,
    elevation_threshold: float = 50.0,
    n_rays: int = 20,
) -> np.ndarray | None:
    # boundary extraction
    road_mask = make_compsite_road_mask(emu, mode, min_id)
    boundary_lines = get_road_lines(emu, road_mask)

    # surface info
    position = emu.memory.driver_position
    mtx = emu.memory.driver_matrix2
    normal = mtx[1]

    # Calculate perpendicular distance to the kart's plane and filter
    n_hat = normal / (np.linalg.norm(normal) + 1e-9)
    vecs = boundary_lines - position
    dists = np.dot(vecs, n_hat)

    # Keep segments where at least one endpoint is within the threshold
    valid_mask = np.min(np.abs(dists), axis=1) < elevation_threshold
    boundary_lines = boundary_lines[valid_mask]

    # Guard against edge cases where the filter removes all lines (e.g., falling out of bounds)
    if len(boundary_lines) == 0:
        return None

    B, T, C = boundary_lines.shape
    boundary_lines = boundary_lines.reshape(B * T, C)

    # ray generation
    ray_origin, ray_direction = generate_plane_vectors(n_rays, 180, mtx, position)

    # projection to local surface
    boundary_lines_uv = project_to_plane(boundary_lines, normal, position)
    boundary_lines_uv = boundary_lines_uv.reshape(B, T, C - 1)
    ray_direction_uv = project_to_plane(ray_direction + position[None, :], normal, position)

    # collect ray intersections
    ray_origin_uv = np.zeros_like(ray_direction_uv)
    t = raycast_2d(boundary_lines_uv, ray_origin_uv, ray_direction_uv).clip(min_dist, max_dist)
    return t
