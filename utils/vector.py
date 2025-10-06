import torch
import torch.nn.functional as F
import math
from utils.emulator import SCREEN_WIDTH, SCREEN_HEIGHT

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
      
def ray_triangle_intersect_mt(
    origins: torch.Tensor,      # (N, 3)
    directions: torch.Tensor,   # (N, 3) -- need not be unit, but 't' scales accordingly
    v0: torch.Tensor,           # (M, 3)
    v1: torch.Tensor,           # (M, 3)
    v2: torch.Tensor,           # (M, 3)
    *,
    eps: float = 1e-8,
    t_min: float = 1e-6,
    t_max: float = float('inf'),
    backface_cull: bool = True,
    return_nearest: bool = True,
    return_barycentric: bool = False,
    return_points: bool = False,
):
    """
    Batched Möller–Trumbore ray/triangle intersections.

    Returns
    -------
    If return_nearest:
        hit_t:      (N,)        smallest positive t per ray (inf if no hit)
        hit_idx:    (N,)        triangle index of the nearest hit (-1 if none)
        [u,v]:      (N,) each   barycentrics (optional; 0 if no hit)
        [points]:   (N,3)       intersection points (optional; 0 if no hit)
        hit_mask:   (N,)        boolean ray-level mask (True if any hit)
    Else (all pairwise):
        t:          (N,M)       t for each ray/triangle (inf if no hit)
        hit_mask:   (N,M)       boolean mask of pairwise hits
        [u,v]:      (N,M) each  barycentrics (optional)
        [points]:   (N,M,3)     points (optional)
    """
    # Ensure all on same device/dtype
    device = origins.device
    dtype = origins.dtype
    directions = directions.to(device=device, dtype=dtype)
    v0 = v0.to(device=device, dtype=dtype)
    v1 = v1.to(device=device, dtype=dtype)
    v2 = v2.to(device=device, dtype=dtype)

    N = origins.shape[0]
    M = v0.shape[0]

    # Triangle edges (M,3)
    e1 = v1 - v0
    e2 = v2 - v0

    # Broadcast shapes:
    # origins, directions: (N,1,3)
    # v0, e1, e2:          (1,M,3)
    O  = origins[:, None, :]
    D  = directions[:, None, :]
    V0 = v0[None, :, :]
    E1 = e1[None, :, :]
    E2 = e2[None, :, :]

    # Möller–Trumbore core
    P = torch.cross(D, E2, dim=-1)                        # (N,M,3)
    a = torch.sum(E1 * P, dim=-1)                         # (N,M)

    if backface_cull:
        # reject parallel or backfacing (a >= -eps)
        denom_ok = a < -eps
    else:
        denom_ok = torch.abs(a) > eps

    f = torch.where(denom_ok, 1.0 / a, torch.zeros_like(a))  # (N,M)

    S = O - V0                                              # (N,M,3)
    u = f * torch.sum(S * P, dim=-1)                        # (N,M)

    u_ok = (u >= 0.0) & (u <= 1.0)

    Q = torch.cross(S, E1, dim=-1)                          # (N,M,3)
    v = f * torch.sum(D * Q, dim=-1)                        # (N,M)
    v_ok = (v >= 0.0) & ((u + v) <= 1.0)

    t = f * torch.sum(E2 * Q, dim=-1)                       # (N,M)
    t_ok = (t >= t_min) & (t <= t_max)

    # Combine masks
    hit_mask = denom_ok & u_ok & v_ok & t_ok                # (N,M)

    # For non-hits, set t to +inf to make reductions easy
    t_out = torch.where(hit_mask, t, torch.full_like(t, float('inf')))

    if not return_nearest:
        outs = {"t": t_out, "hit_mask": hit_mask}
        if return_barycentric:
            outs["u"] = torch.where(hit_mask, u, torch.zeros_like(u))
            outs["v"] = torch.where(hit_mask, v, torch.zeros_like(v))
        if return_points:
            Pts = O + t_out[..., None] * D                  # (N,M,3) with inf*D -> inf; mask it
            Pts = torch.where(hit_mask[..., None], Pts, torch.zeros_like(Pts))
            outs["points"] = Pts
        return outs

    # Nearest hit per ray
    # argmin over M; if all inf -> no hit
    hit_t, hit_idx = torch.min(t_out, dim=1)                # (N,), (N,)
    any_hit = torch.isfinite(hit_t)

    # Make a clean hit_idx: -1 where no hit
    hit_idx = torch.where(any_hit, hit_idx, torch.full_like(hit_idx, -1))

    result = {
        "t": hit_t,             # (N,)
        "tri_idx": hit_idx,     # (N,)
        "hit_mask": any_hit,    # (N,)
    }

    if return_barycentric or return_points:
        # Gather per-ray u, v at the chosen triangle
        # To avoid index errors when no hit, clamp idx to 0 then mask later
        safe_idx = torch.clamp(hit_idx, min=0)
        gather_idx = safe_idx[:, None]                       # (N,1)

        if return_barycentric:
            u_near = u.gather(1, gather_idx).squeeze(1)      # (N,)
            v_near = v.gather(1, gather_idx).squeeze(1)      # (N,)
            u_near = torch.where(any_hit, u_near, torch.zeros_like(u_near))
            v_near = torch.where(any_hit, v_near, torch.zeros_like(v_near))
            result["u"] = u_near
            result["v"] = v_near

        if return_points:
            # Intersection point: O + t*D
            pts = O[:, 0, :] + hit_t[:, None] * D[:, 0, :]
            pts = torch.where(any_hit[:, None], pts, torch.zeros_like(pts))
            result["points"] = pts

    return result
        
def triangle_raycast_batch(
    ray_origin: torch.Tensor,  # (N, 3)
    ray_dir: torch.Tensor,     # (N, 3)
    v1: torch.Tensor,          # (M, 3)
    v2: torch.Tensor,          # (M, 3)
    v3: torch.Tensor,          # (M, 3)
    epsilon=1e-8
) -> torch.Tensor:
    """
    Batched ray-triangle intersection using Möller–Trumbore.
    
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
    
    # Mask invalid with NaN
    pts = pts[None, valid]

    return pts
    

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
    
def project_to_screen(world_points, model_view, fov, aspect, far, near, device=None):
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
    
    ndc = clip_space[:, :2] / clip_space[:, 3, None]
    
    screen_x = (ndc[:, 0] + 1) / 2 * SCREEN_WIDTH
    screen_y = (1 - ndc[:, 1]) / 2 * SCREEN_HEIGHT
    screen_depth = clip_space[:, 2]
    return torch.stack([screen_x, screen_y, screen_depth], dim=-1)

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