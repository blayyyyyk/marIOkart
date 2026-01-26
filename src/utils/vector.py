from __future__ import annotations
import torch
import torch.nn.functional as F
import math
import numpy as np
from typing import Union
#from utils.emulator import SCREEN_WIDTH, SCREEN_HEIGHT

SCREEN_WIDTH, SCREEN_HEIGHT = 256, 192

def get_mps_device() -> torch.device:
    # Check that MPS is available
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print(
                "MPS not available because the current PyTorch install was not "
                "built with MPS enabled."
            )
        else:
            print(
                "MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine."
            )

        exit()

    device = torch.device("mps")
    return device


def _determinant_2d(x0, y0, x1, y1):
    return x0 * y1 - y0 * x1


def cross_product_2d(p0: torch.Tensor, p1: torch.Tensor) -> torch.Tensor:
    return _determinant_2d(p0[..., 0], p0[..., 1], p1[..., 0], p1[..., 1])


def triangle_raycast(
    ray_origin: torch.Tensor,
    ray_dir: torch.Tensor,
    v1: torch.Tensor,
    v2: torch.Tensor,
    v3: torch.Tensor,
    epsilon=1e-8
) -> torch.Tensor | None:
    # Test if the ray intersects the triangle
    edge1 = v2 - v1
    edge2 = v3 - v1
    ray_cross_e2 = torch.cross(ray_dir, edge2)
    det = torch.dot(edge1, ray_cross_e2)

    if det > -epsilon and det < epsilon:
        return None

    inv_det = 1.0 / det
    s = ray_origin - v1
    u = inv_det * torch.dot(s, ray_cross_e2)

    if (u < 0 and torch.abs(u) > epsilon) or (u > 1 and torch.abs(u - 1) > epsilon):
        return None

    s_cross_e1 = torch.dot(s, edge1)
    v = inv_det * torch.dot(ray_dir, s_cross_e1)

    if (v < 0 and torch.abs(v) > epsilon) or (u + v > 1 and torch.abs(u + v - 1) > epsilon):
        return None

    # Compute where the ray intersects the triangle
    t = inv_det * torch.dot(edge2, s_cross_e1)
    if t > epsilon:
        return ray_origin + ray_dir * t
    else:
        return None

import torch

def intersect_ray_line_2d(O, D, P1, P2, eps=1e-8):
    """
    Find intersection of a ray (O + tD, t>=0)
    with a line segment between P1 and P2.

    All inputs are torch tensors of shape (..., 2)
    and can be batched.
    """
    # Direction of the segment
    v = P2 - P1

    # 2D cross product helper (scalar)
    def cross(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    # Compute determinant
    denom = cross(D, v)  # parallel if denom == 0

    # Compute relative position
    w = P1 - O

    t = cross(w, v) / (denom + eps)
    u = cross(w, D) / (denom + eps)

    # Compute intersection point
    intersection = O + t.unsqueeze(-1) * D

    # Valid if:
    valid = (denom.abs() > eps) & (t >= 0) & (u >= 0) & (u <= 1)

    return intersection, valid

def triangle_raycast_batch(
    ray_origin: torch.Tensor, # (N, 3)
    ray_dir: torch.Tensor, # (N, 3)
    v1: torch.Tensor, # (M, 3)
    v2: torch.Tensor, # (M, 3)
    v3: torch.Tensor, # (M, 3)
    epsilon=1e-8
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Batched ray-triangle intersection using Moller–Trumbore.

    Args:
        ray_origin: (N, 3) origins
        ray_dir: (N, 3) directions (normalized)
        v1, v2, v3: (M, 3) triangle vertices
    Returns:
        intersections: (N, M, 3) intersection points (NaN if no hit)
    """
    # Expand to (N, M, 3)
    ro = ray_origin[:, None, :]   # (N,1,3)
    rd = ray_dir[:, None, :]      # (N,1,3)
    v1 = v1[None, :, :]           # (1,M,3)
    v2 = v2[None, :, :]
    v3 = v3[None, :, :]

    edge1 = v2 - v1
    edge2 = v3 - v1

    h = torch.cross(rd, edge2, dim=-1)  # (N,M,3)
    a = torch.sum(edge1 * h, dim=-1)    # (N,M)

    mask_parallel = (a.abs() < epsilon)

    f = 1.0 / (a + epsilon * mask_parallel.sign())  # safe div
    s = ro - v1
    u = f * torch.sum(s * h, dim=-1)

    mask_u = (u < 0) | (u > 1)

    q = torch.cross(s, edge1, dim=-1)
    v = f * torch.sum(rd * q, dim=-1)
    mask_v = (v < 0) | (u + v > 1)

    t = f * torch.sum(edge2 * q, dim=-1)
    mask_t = (t <= epsilon)

    # Valid hits
    valid = ~(mask_parallel | mask_u | mask_v | mask_t)

    # Intersection points
    pts = ro + rd * t[..., None]  # (N,M,3)

    return pts, valid

def generate_driver_rays(
    n_rays: int, 
    sweep_angle: float, 
    driver_pos: torch.Tensor, 
    driver_fwd: torch.Tensor, 
    driver_up: torch.Tensor
):
    """
    Generates n rays centered on the driver's forward vector, spread across the XZ plane (driver space).
    
    Args:
        n_rays:      Number of rays to generate.
        sweep_angle: Total field of view in degrees (e.g., 120 means +/- 60 degrees).
        driver_pos:  (3,) Global position of the driver.
        driver_fwd:  (3,) Global forward vector (normalized).
        driver_up:   (3,) Global up vector (normalized).
        
    Returns:
        ray_origins:    (n_rays, 3) Global starting points.
        ray_directions: (n_rays, 3) Global normalized direction vectors.
    """
    device = driver_pos.device
    
    # --- STEP 1: DEFINE RAYS IN LOCAL SPACE ---
    # We define the pattern relative to a car at (0,0,0) facing (0,0,1).
    # X = Right, Y = Up, Z = Forward
    
    # Generate angles centered at 0 (Forward)
    # If sweep is 90, we go from -45 to +45.
    half_sweep = sweep_angle / 2.0
    angles_deg = torch.linspace(-half_sweep, half_sweep, n_rays, device=device)
    angles_rad = torch.deg2rad(angles_deg)
    
    # Convert Polar -> Cartesian (Local XZ Plane)
    # x = sin(theta) (Positive is Right)
    # y = 0          (Flat on the driver's horizon)
    # z = cos(theta) (Positive is Forward)
    local_x = torch.sin(angles_rad)
    local_y = torch.zeros_like(angles_rad)
    local_z = torch.cos(angles_rad)
    
    # Shape: (n_rays, 3)
    local_dirs = torch.stack([local_x, local_y, local_z], dim=1)

    # --- STEP 2: BUILD ROTATION MATRIX (Local -> World) ---
    # We need the 3rd basis vector: Right.
    # Right = Cross(Forward, Up).
    # Note: We assume standard Right-Hand Rule.
    driver_right = torch.cross(driver_fwd, driver_up)
    
    # Normalize just in case inputs weren't perfect
    driver_right = driver_right / (torch.linalg.norm(driver_right) + 1e-6)
    
    # Construct the Rotation Matrix
    # Columns are [Right, Up, Forward]
    # Shape: (3, 3)
    rotation_matrix = torch.stack([driver_right, driver_up, driver_fwd], dim=1)
    
    # --- STEP 3: TRANSFORM TO GLOBAL SPACE ---
    # Rotate: (N, 3) @ (3, 3)^T
    # We transpose because standard matrix math is R @ v, but our vectors are rows.
    global_dirs = local_dirs @ rotation_matrix.T
    
    # Translate: Add driver position
    # Expand driver_pos to match (n_rays, 3)
    global_origins = driver_pos.expand(n_rays, 3)
    
    return global_origins, global_dirs

def pairwise_distances(pts):
    norms = (pts**2).sum(dim=-1)
    G = pts @ pts.T
    D2 = norms[:, None] + norms[None, :] - 2 * G
    D2 = torch.clamp(D2, min=0.0)
    D = torch.sqrt(D2)
    return D

def pairwise_distances_cross(A, B):
    # Compute squared norms
    A_norms = (A**2).sum(dim=1).unsqueeze(1)  # n x 1
    B_norms = (B**2).sum(dim=1).unsqueeze(0)  # 1 x m

    # Compute pairwise squared distances
    D_squared = A_norms + B_norms - 2 * (A @ B.T)

    # Euclidean distances
    D = torch.sqrt(D_squared.clamp(min=0))
    return D

def compute_orthonormal_basis(forward_vector_3d, reference_vector_3d=None, device=None):
    if reference_vector_3d is None:
        reference_vector_3d = torch.tensor([0., 1., 0.],
            dtype=forward_vector_3d.dtype,
            device=forward_vector_3d.device)

    right_vector_3d = torch.cross(forward_vector_3d, reference_vector_3d, dim=0)
    right_vector_3d /= right_vector_3d.norm()

    up_vector_3d = torch.cross(right_vector_3d, forward_vector_3d, dim=0)
    up_vector_3d /= up_vector_3d.norm()

    basis = torch.stack([
        right_vector_3d,
        up_vector_3d,
        forward_vector_3d,
    ], dim=0)

    return basis

def compute_model_view(camera_pos, camera_target_pos, device=None):
    forward = camera_target_pos - camera_pos
    forward /= torch.norm(forward, dim=-1)

    rot = compute_orthonormal_basis(forward, device=device)

    pos_proj = rot @ camera_pos.unsqueeze(-2).transpose(-1, -2)

    model_view = torch.eye(4, dtype=rot.dtype, device=device)
    model_view[:3,:3] = rot
    model_view[:3,3] = -pos_proj.squeeze(-1)

    return model_view

def project_to_screen(world_points, model_view, fov, aspect, far, near, z_scale, device=None):
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
    projection_matrix[2, 2] = (far + near) / (near - far)
    projection_matrix[2, 3] = -(2 * far * near) / (near - far)
    projection_matrix[3, 2] = 1

    clip_space = (projection_matrix @ cam_space.T).T

    ndc = clip_space[:, :3] / clip_space[:, 3, None]

    screen_x = (ndc[:, 0] + 1) / 2 * SCREEN_WIDTH
    screen_y = (1 - ndc[:, 1]) / 2 * SCREEN_HEIGHT
    screen_depth = clip_space[:, 2]
    screen_depth_norm = -far / (-far + z_scale * clip_space[:, 2])
    return torch.stack([screen_x, screen_y, screen_depth, screen_depth_norm], dim=-1)

def sample_cone(x, theta, k):
    device = x.device
    dtype = x.dtype
    phi = torch.rand(k, device=device) * torch.pi * 2

    cos_alpha = torch.rand(k, device=device) * ((1 - math.cos(theta)) + math.cos(theta))
    alpha = torch.arccos(cos_alpha)
    sin_alpha = torch.sin(alpha)
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    local = torch.stack([
        sin_alpha * cos_phi,
        sin_alpha * sin_phi,
        cos_alpha
    ], dim=-1)

    if torch.abs(x[2]) < 0.99:
        ref = torch.tensor([0.,0.,1.], device=device, dtype=dtype)
    else:
        ref = torch.tensor([0.,1.,0.], device=device, dtype=dtype)

    u = torch.nn.functional.normalize(torch.cross(ref, x, dim=0), dim=0)
    v = torch.cross(x, u)

    # Basis matrix
    basis = torch.stack([u, v, x], dim=1)  # (3,3)

    # Step 3: rotate local vectors into world space
    world = local @ basis.T
    return world


def sample_circular_sweep(
    forward: torch.Tensor,
    left: torch.Tensor,
    up: torch.Tensor,
    sweep_plane: str = "xz",
    interval: tuple[float, float] = (-0.5, 0.5),
    n_steps: int = 32,
):
    """
    Generate a semicircular sweep of direction vectors across a 2d-subspace (i.e. plane).

    Args:
        forward (torch.Tensor): (3,) normalized forward vector.
        left (torch.Tensor): (3,) normalized left vector.
        up (torch.Tensor): (3,) normalized up vector (unused except for completeness).
        sweep_plane (str): the sweeping plane ()
        interval (float): angle difference between each sample in radians.
        n_steps (int): Number of vectors in the sweep.

    Returns:
        torch.Tensor: (n_steps, 3) array of direction vectors.
    """
    # Ensure all inputs are normalized and shaped properly
    forward = forward / forward.norm()
    left = left / left.norm()
    up = up / up.norm()  # included for consistency even if unused
    
    if sweep_plane == 'xy':
        a_sin = up
        a_cos = left
    elif sweep_plane == 'yz':
        a_sin = forward
        a_cos = up
    elif sweep_plane == 'xz':
        a_sin = left
        a_cos = forward
        
    else:
        raise ValueError(f"Invalid sweep plane: {sweep_plane}")


    # Sweep angles from -90° to +90° (front half)
    theta = torch.linspace(interval[0] * torch.pi, interval[1] * torch.pi, n_steps, device=forward.device)

    # Compute direction vectors (in the forward-left plane)
    dirs = (torch.cos(theta).unsqueeze(1) * a_cos +
            torch.sin(theta).unsqueeze(1) * a_sin)

    # Normalize all output vectors
    dirs = dirs / dirs.norm(dim=1, keepdim=True)

    return dirs


def sample_semicircular_sweep(
    forward: torch.Tensor,
    left: torch.Tensor,
    up: torch.Tensor,
    interval: tuple[float, float] = (-0.5, 0.5),
    n_steps: int = 32
) -> torch.Tensor:
    """
    Generate a semicircular sweep of direction vectors across the front hemisphere.

    Args:
        forward (torch.Tensor): (3,) normalized forward vector.
        left (torch.Tensor): (3,) normalized left vector.
        up (torch.Tensor): (3,) normalized up vector (unused except for completeness).
        n_steps (int): Number of vectors in the sweep.

    Returns:
        torch.Tensor: (n_steps, 3) array of direction vectors.
    """
    # Ensure all inputs are normalized and shaped properly
    forward = forward / forward.norm()
    left = left / left.norm()
    up = up / up.norm()  # included for consistency even if unused

    # Sweep angles from -90° to +90° (front half)
    theta = torch.linspace(interval[0] * torch.pi, interval[1] * torch.pi, n_steps, device=forward.device)

    # Compute direction vectors (in the forward-left plane)
    dirs = (torch.cos(theta).unsqueeze(1) * forward +
            torch.sin(theta).unsqueeze(1) * left)

    # Normalize all output vectors
    dirs = dirs / dirs.norm(dim=1, keepdim=True)

    return dirs

def clipped_mean(points: torch.Tensor, std_thresh: float = 2.0):
    """
    Compute mean of 3D points, keeping only those within `std_thresh` stds
    of the mean.

    points: (N,3) tensor
    std_thresh: threshold in standard deviations
    """
    # First-pass mean and std (over points, separately for x/y/z)
    mean = points.mean(dim=0, keepdim=True)    # (1,3)
    std  = points.std(dim=0, unbiased=False, keepdim=True)  # (1,3)

    # Compute z-scores (absolute deviations scaled by std)
    z = (points - mean).abs() / (std + 1e-8)   # (N,3)

    # A point is valid if *all* coords are within the threshold
    mask = (z <= std_thresh).all(dim=1)        # (N,)

    # Filter points
    valid_points = points[mask]

    if valid_points.numel() == 0:
        # fallback: return global mean if all filtered out
        return mean.squeeze(0), mask

    return valid_points.mean(dim=0), mask


def interpolate(x0, x1, alpha):
    return (1 - alpha) * x0 + (alpha * x1)

def smooth_mean(x0: torch.Tensor, sx1: torch.Tensor, alpha: float, std_threshold: float):
    x1, _ = clipped_mean(sx1, std_threshold)
    return interpolate(x0, x1, alpha)

def project(a: torch.Tensor, b: torch.Tensor):
    x = a.dot(b)
    y = b.dot(b)
    return x / y * b

def extrapolate(P: torch.Tensor, source: torch.Tensor, dim=0):
    dim_mask = torch.range(0, source.shape[1] - 1, 1) != dim

    D = pairwise_distances_cross(P, source[:, dim_mask])
    min_idx = D.argmin(dim=1)

    result = torch.ones(P.shape[0], P.shape[1] + 1, device=P.device)
    result[:, dim_mask] = P
    result[:, dim] = source[min_idx, dim]

    return result

def triangle_altitude(a, b):
    assert isinstance(a, torch.Tensor) == isinstance(b, torch.Tensor)

    if isinstance(a, torch.Tensor):
        return a * b / torch.sqrt(a**2 + b**2)

    return a * b / math.sqrt(a**2 + b**2)

import torch

def generate_plane_vectors(
    n_rays: int,
    sweep_angle_deg: float,
    rotation_matrix: torch.Tensor,
    origin: torch.Tensor,
    plane_indices: tuple[int, int] = (2, 0) # Default: Forward (Z) to Right (X)
):
    """
    Generates a fan of rays using a rotation matrix to define the orientation.
    
    Args:
        n_rays: Number of rays to generate.
        sweep_angle_deg: Total field of view (e.g., 120 means +/- 60 deg).
        rotation_matrix: (3, 3) or (B, 3, 3) Matrix where columns are basis vectors.
                         - Col 0: Right (X)
                         - Col 1: Up (Y)
                         - Col 2: Forward (Z)
        origin: (3,) or (B, 3) Ray starting positions.
        plane_indices: Tuple (center_idx, side_idx) defining the sweep plane.
                       - (2, 0) = Horizontal Sweep (Forward -> Right)
                       - (2, 1) = Vertical Sweep (Forward -> Up)
        
    Returns:
        origins: (N, 3) or (B, N, 3)
        directions: (N, 3) or (B, N, 3) Normalized
    """
    # 1. Handle Batching
    # If input is unbatched (3,3), unsqueeze to (1,3,3) for uniform logic
    is_batched = rotation_matrix.dim() == 3
    if not is_batched:
        rotation_matrix = rotation_matrix.unsqueeze(0)
        origin = origin.unsqueeze(0)
        
    B = rotation_matrix.shape[0]
    device = rotation_matrix.device
    
    # 2. Extract Basis Vectors from Matrix Columns
    # R[:, :, i] grabs the i-th column (the i-th basis vector)
    center_idx, side_idx = plane_indices
    
    # Shape: (B, 3)
    vec_center = rotation_matrix[:, :, center_idx] 
    vec_side   = rotation_matrix[:, :, side_idx]
    
    # 3. Generate Angles
    half_sweep = sweep_angle_deg / 2.0
    angles = torch.linspace(-half_sweep, half_sweep, n_rays, device=device)
    rads = torch.deg2rad(angles)
    
    # 4. Reshape for Broadcasting
    # We want: (B, 1, 3) * (1, N, 1) -> (B, N, 3)
    
    # Basis vectors: (B, 1, 3)
    vec_center = vec_center.unsqueeze(1)
    vec_side   = vec_side.unsqueeze(1)
    
    # Trig values: (1, N, 1)
    cos_t = torch.cos(rads).view(1, n_rays, 1)
    sin_t = torch.sin(rads).view(1, n_rays, 1)
    
    # 5. Linear Combination (The Sweep)
    # v = Center * cos(t) + Side * sin(t)
    directions = (vec_center * cos_t) + (vec_side * sin_t)
    
    # Normalize (just in case floating point drift occurred)
    directions = directions / (torch.norm(directions, dim=-1, keepdim=True) + 1e-8)
    
    # 6. Expand Origins
    # Origin (B, 3) -> (B, N, 3)
    origins = origin.unsqueeze(1).expand(-1, n_rays, -1)
    
    # 7. Remove batch dim if input was single
    if not is_batched:
        return origins.squeeze(0), directions.squeeze(0)
        
    return origins, directions
    
def ray_triangle_intersection(
    vertices: torch.Tensor, 
    ray_origin: torch.Tensor, 
    ray_direction: torch.Tensor, 
    culling: bool = False, 
    epsilon: float = 1e-6
) -> torch.Tensor:
    """
    PyTorch vmap-compatible Möller-Trumbore intersection.
    
    Args:
        vertices: (3, 3) Tensor
        ray_origin: (3,) Tensor
        ray_direction: (3,) Tensor
        culling: bool (Static python argument, not a Tensor)
        epsilon: float
        
    Returns:
        Tensor of shape (3,) containing [t, u, v]. 
        Contains [NaN, NaN, NaN] if no intersection occurs.
    """
    v0, v1, v2 = vertices[0], vertices[1], vertices[2]

    edge1 = v1 - v0
    edge2 = v2 - v0

    pvec = torch.linalg.cross(ray_direction, edge2)

    det = torch.dot(edge1, pvec)

    if culling:
        # Culling: determinant must be positive and > epsilon
        valid_det = det > epsilon
    else:
        # No Culling: determinant must be non-zero (abs > epsilon)
        valid_det = torch.abs(det) > epsilon

    safe_det = torch.where(valid_det, det, torch.ones_like(det)) 
    inv_det = 1.0 / safe_det

    tvec = ray_origin - v0
    u = torch.dot(tvec, pvec) * inv_det
    valid_u = (u >= 0.0) & (u <= 1.0)

    qvec = torch.linalg.cross(tvec, edge1)
    v = torch.dot(ray_direction, qvec) * inv_det
    valid_v = (v >= 0.0) & (u + v <= 1.0)

    t = torch.dot(edge2, qvec) * inv_det
    valid_t = t > epsilon # Intersection must be strictly in front of camera

    is_hit = valid_det & valid_u & valid_v & valid_t

    result = torch.stack([t, u, v])
    
    nan_tensor = torch.full_like(result, float('nan'))

    return torch.where(is_hit, result, nan_tensor)
    
