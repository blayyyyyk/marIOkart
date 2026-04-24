from __future__ import annotations

import math

import numpy as np
import torch

# from utils.emulator import SCREEN_WIDTH, SCREEN_HEIGHT

SCREEN_WIDTH, SCREEN_HEIGHT = 256, 192


def get_available_devices() -> list[str]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append(f"cuda:{i}")

    if torch.backends.mps.is_available():
        devices.append("mps")

    # NOTE: project is untested on this device type
    if torch.xpu.is_available():
        for i in range(torch.xpu.device_count()):
            devices.append(f"xpu:{i}")

    return devices


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
    epsilon=1e-8,
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


def triangle_raycast_batch(
    ray_origin: torch.Tensor,  # (N, 3)
    ray_dir: torch.Tensor,  # (N, 3)
    v1: torch.Tensor,  # (M, 3)
    v2: torch.Tensor,  # (M, 3)
    v3: torch.Tensor,  # (M, 3)
    epsilon=1e-8,
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
    ro = ray_origin[:, None, :]  # (N,1,3)
    rd = ray_dir[:, None, :]  # (N,1,3)
    v1 = v1[None, :, :]  # (1,M,3)
    v2 = v2[None, :, :]
    v3 = v3[None, :, :]

    edge1 = v2 - v1
    edge2 = v3 - v1

    h = torch.cross(rd, edge2, dim=-1)  # (N,M,3)
    a = torch.sum(edge1 * h, dim=-1)  # (N,M)

    mask_parallel = a.abs() < epsilon

    f = 1.0 / (a + epsilon * mask_parallel.sign())  # safe div
    s = ro - v1
    u = f * torch.sum(s * h, dim=-1)

    mask_u = (u < 0) | (u > 1)

    q = torch.cross(s, edge1, dim=-1)
    v = f * torch.sum(rd * q, dim=-1)
    mask_v = (v < 0) | (u + v > 1)

    t = f * torch.sum(edge2 * q, dim=-1)
    mask_t = t <= epsilon

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
    driver_up: torch.Tensor,
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


def compute_model_view(camera_pos, camera_target_pos, device=None):
    forward = camera_target_pos - camera_pos
    forward /= torch.norm(forward, dim=-1)

    rot = compute_orthonormal_basis(forward, device=device)

    pos_proj = rot @ camera_pos.unsqueeze(-2).transpose(-1, -2)

    model_view = torch.eye(4, dtype=rot.dtype, device=device)
    model_view[:3, :3] = rot
    model_view[:3, 3] = -pos_proj.squeeze(-1)

    return model_view


def sample_cone(x, theta, k):
    device = x.device
    dtype = x.dtype
    phi = torch.rand(k, device=device) * torch.pi * 2

    cos_alpha = torch.rand(k, device=device) * ((1 - math.cos(theta)) + math.cos(theta))
    alpha = torch.arccos(cos_alpha)
    sin_alpha = torch.sin(alpha)
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    local = torch.stack([sin_alpha * cos_phi, sin_alpha * sin_phi, cos_alpha], dim=-1)

    if torch.abs(x[2]) < 0.99:
        ref = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    else:
        ref = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)

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

    if sweep_plane == "xy":
        a_sin = up
        a_cos = left
    elif sweep_plane == "yz":
        a_sin = forward
        a_cos = up
    elif sweep_plane == "xz":
        a_sin = left
        a_cos = forward

    else:
        raise ValueError(f"Invalid sweep plane: {sweep_plane}")

    # Sweep angles from -90° to +90° (front half)
    theta = torch.linspace(interval[0] * torch.pi, interval[1] * torch.pi, n_steps, device=forward.device)

    # Compute direction vectors (in the forward-left plane)
    dirs = torch.cos(theta).unsqueeze(1) * a_cos + torch.sin(theta).unsqueeze(1) * a_sin

    # Normalize all output vectors
    dirs = dirs / dirs.norm(dim=1, keepdim=True)

    return dirs


def sample_semicircular_sweep(
    forward: torch.Tensor,
    left: torch.Tensor,
    up: torch.Tensor,
    interval: tuple[float, float] = (-0.5, 0.5),
    n_steps: int = 32,
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
    dirs = torch.cos(theta).unsqueeze(1) * forward + torch.sin(theta).unsqueeze(1) * left

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
    mean = points.mean(dim=0, keepdim=True)  # (1,3)
    std = points.std(dim=0, unbiased=False, keepdim=True)  # (1,3)

    # Compute z-scores (absolute deviations scaled by std)
    z = (points - mean).abs() / (std + 1e-8)  # (N,3)

    # A point is valid if *all* coords are within the threshold
    mask = (z <= std_thresh).all(dim=1)  # (N,)

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


def generate_plane_vectors(
    n_rays: int,
    sweep_angle_deg: float,
    rotation_matrix: np.ndarray,
    origin: np.ndarray,
    plane_indices: tuple[int, int] = (2, 0),  # Default: Forward (Z) to Right (X)
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
    is_batched = rotation_matrix.ndim == 3
    if not is_batched:
        rotation_matrix = rotation_matrix[None, ...]
        origin = origin[None, ...]

    # 2. Extract Basis Vectors from Matrix Columns
    # R[:, :, i] grabs the i-th column (the i-th basis vector)
    center_idx, side_idx = plane_indices

    # Shape: (B, 3)
    vec_center = rotation_matrix[:, :, center_idx]
    vec_side = rotation_matrix[:, :, side_idx]

    # 3. Generate Angles
    half_sweep = sweep_angle_deg / 2.0
    angles = np.linspace(-half_sweep, half_sweep, n_rays)
    rads = np.deg2rad(angles)

    # 4. Reshape for Broadcasting
    # We want: (B, 1, 3) * (1, N, 1) -> (B, N, 3)

    # Basis vectors: (B, 1, 3)
    vec_center = vec_center[:, None, ...]
    vec_side = vec_side[:, None, ...]

    # Trig values: (1, N, 1)
    cos_t = np.cos(rads).reshape(1, n_rays, 1)
    sin_t = np.sin(rads).reshape(1, n_rays, 1)

    # 5. Linear Combination (The Sweep)
    # v = Center * cos(t) + Side * sin(t)
    directions = (vec_center * cos_t) + (vec_side * sin_t)

    # Normalize (just in case floating point drift occurred)
    directions = directions / (np.linalg.norm(directions, axis=-1, keepdims=True) + 1e-8)

    # 6. Expand Origins
    # Origin (B, 3) -> (B, N, 3)
    origins = origin[:, None, ...].repeat(n_rays, axis=1)

    # 7. Remove batch dim if input was single
    if not is_batched:
        return origins.squeeze(0), directions.squeeze(0)

    return origins, directions


def ray_triangle_intersection(
    vertices: torch.Tensor,
    ray_origin: torch.Tensor,
    ray_direction: torch.Tensor,
    culling: bool = False,
    epsilon: float = 1e-6,
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
    valid_t = t > epsilon  # Intersection must be strictly in front of camera

    is_hit = valid_det & valid_u & valid_v & valid_t

    result = torch.stack([t, u, v])

    nan_tensor = torch.full_like(result, float("nan"))

    return torch.where(is_hit, result, nan_tensor)


def project_to_plane(points: np.ndarray, normal: np.ndarray, plane_point: np.ndarray | None = None):
    """
    Projects an (N, 3) array of points onto a plane.

    Args:
        points: np.ndarray of shape (N, 3).
        normal: np.ndarray of shape (3,) representing the plane's normal vector.
        plane_point: np.ndarray of shape (3,) for a point on the plane. Defaults to origin.

    Returns:
        projected_3d: (N, 3) array of the points flattened onto the plane.
        local_2d: (N, 2) array of the points in the plane's local coordinate system.
    """
    if plane_point is None:
        plane_point = np.zeros(3)

    # normalize
    n_hat = normal / np.linalg.norm(normal)
    v = points - plane_point  # compute delta
    dists = np.dot(v, n_hat)
    projected_3d = points - (dists[:, np.newaxis] * n_hat)
    axis_idx = np.argmin(np.abs(n_hat))
    arbitrary_vec = np.zeros(3)
    arbitrary_vec[axis_idx] = 1.0

    # create orthoganol basis
    u_vec = np.cross(n_hat, arbitrary_vec)
    u_vec /= np.linalg.norm(u_vec)
    v_vec = np.cross(n_hat, u_vec)  # this is assumed orthoganol...

    # project onto plane, uv coords
    vectors_on_plane = projected_3d - plane_point
    u_coords = np.dot(vectors_on_plane, u_vec)
    v_coords = np.dot(vectors_on_plane, v_vec)
    local_2d = np.column_stack((u_coords, v_coords))
    return local_2d


def raycast_2d(segments: np.ndarray, ray_origins: np.ndarray, ray_dirs: np.ndarray) -> np.ndarray:
    """
    Casts N rays against M line segments and returns the minimum t distance for each ray.

    Args:
        segments: (M, 2, 2) array of line segments [Endpoint_A, Endpoint_B].
        ray_origins: (N, 2) or (2,) array of ray start points.
        ray_dirs: (N, 2) array of ray direction vectors.

    Returns:
        min_t: (N,) array of distances to the closest segment. Misses are marked as np.inf.
    """
    # split segment endpoints
    A = segments[:, 0, :]  # (M, 2)
    B = segments[:, 1, :]
    S = B - A

    # conform ray shapes
    ray_dirs = np.atleast_2d(ray_dirs)
    ray_origins = np.atleast_2d(ray_origins)
    if len(ray_origins) == 1 and len(ray_dirs) > 1:
        ray_origins = np.broadcast_to(ray_origins, ray_dirs.shape)

    P = ray_origins
    R = ray_dirs
    P_exp = P[:, np.newaxis, :]  # (N, 1, 2)
    R_exp = R[:, np.newaxis, :]  # (N, 1, 2)
    A_exp = A[np.newaxis, :, :]  # (1, M, 2)
    S_exp = S[np.newaxis, :, :]  # (1, M, 2)

    delta = A_exp - P_exp  # (N, M, 2)

    def cross_2d(v, w):
        return v[..., 0] * w[..., 1] - v[..., 1] * w[..., 0]

    den = cross_2d(R_exp, S_exp)  # (N, M)
    num_t = cross_2d(delta, S_exp)  # (N, M)
    num_u = cross_2d(delta, R_exp)  # (N, M)

    # solve for t and u (we only need t tho)
    old_settings = np.seterr(divide="ignore", invalid="ignore")  # ignore stupid error
    t = num_t / den
    u = num_u / den
    np.seterr(**old_settings)

    # intersection rules
    epsilon = 1e-8
    valid_hits = (np.abs(den) > epsilon) & (t >= 0) & (u >= 0) & (u <= 1)

    # invalid hit mask
    t_valid = np.where(valid_hits, t, np.inf)

    # return minimum distance ray
    if t_valid.size == 0:
        return np.full_like(ray_origins[:, 0], np.inf)

    min_t = np.min(t_valid, axis=1)  # (N,)
    return min_t
