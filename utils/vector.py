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


def raycast_2d(pos, dir, p0, p1):
    p_delta = p1 - p0
    p_offset = p0 - pos
    cross_bottom = cross_product_2d(dir, p_delta)
    # Calculate the length of the ray
    t = cross_product_2d(p_offset, p_delta) / cross_bottom
    # Calculate the position of the ray's intersection along the line segment between p0 and p1
    u = cross_product_2d(p_offset, dir) / cross_bottom
    # NOTE: For the ray to be pointing towards p0 and p1, t >= 0 and 0 <= u <= 1
    return t, u


def get_rays_2d(pos_2d: torch.Tensor, dir_2d: torch.Tensor, pts_2d: torch.Tensor):
    D = torch.cdist(pts_2d, pts_2d)
    D.fill_diagonal_(float("inf"))

    D[D == 0.0] = float("inf")
    _, indices = torch.topk(D, 2, largest=False)
    neighbors = pts_2d[indices]
    n0, n1 = neighbors.chunk(2, dim=1)
    segment_0 = torch.cat([pts_2d.unsqueeze(1), n0], dim=1)
    segment_1 = torch.cat([pts_2d.unsqueeze(1), n1], dim=1)
    segments = torch.cat([segment_0, segment_1], dim=0)

    p0, p1 = segments.chunk(2, dim=1)
    p0 = p0.squeeze(1)
    p1 = p1.squeeze(1)

    pos_2d = pos_2d.unsqueeze(0).repeat(segments.shape[0], 1)
    dir_2d = dir_2d.unsqueeze(0).repeat(segments.shape[0], 1)
    t, u = raycast_2d(pos_2d, dir_2d, p0, p1)

    valid_mask = (u >= 0) & (u <= 1) & (t >= 0)

    return t[valid_mask], u[valid_mask]


def pairwise_distances(pts):
    norms = (pts**2).sum(dim=-1)
    G = pts @ pts.T
    D2 = norms[:, None] + norms[None, :] - 2 * G
    D2 = torch.clamp(D2, min=0.0)
    D = torch.sqrt(D2)
    return D

def project_to_screen(point_4d: torch.Tensor, proj_4d: torch.Tensor, screen_width: int, screen_height: int) -> torch.Tensor:
    if len(point_4d.shape) == 1:
        point_4d = point_4d.unsqueeze(0)

    # Project point

    point_4d = point_4d.unsqueeze(-2)

    point_proj = proj_4d @ point_4d.transpose(-1, -2)
    point_proj = point_proj.transpose(-1, -2).squeeze(-2)

    point_proj_norm = point_proj / point_proj[:, 3, None] # Normalize

    # Convert to screen space coordinated
    screen_x = (point_proj_norm[:, 0] + 1) / 2 * screen_width
    screen_y = (1 - point_proj_norm[:, 1]) / 2 * screen_height
    screen_proj_point = torch.cat([screen_x[:, None], screen_y[:, None]], dim=-1)

    return screen_proj_point

def compute_orthonormal_basis(forward_vector_3d, reference_vector_3d=None, device=None):
    if reference_vector_3d is None:
        reference_vector_3d = torch.tensor([0., 1., 0.],
            dtype=forward_vector_3d.dtype,
            device=forward_vector_3d.device)

    right_vector_3d = torch.cross(reference_vector_3d, forward_vector_3d, dim=0)
    right_vector_3d /= right_vector_3d.norm()

    up_vector_3d = torch.cross(forward_vector_3d, right_vector_3d, dim=0)
    up_vector_3d /= up_vector_3d.norm()

    basis = torch.stack([
        right_vector_3d,
        up_vector_3d,
        forward_vector_3d,
    ], dim=0) # row-major rotation matrix

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
    
def project_to_camera(world_points, model_view, fov, aspect, device=None):
    N = world_points.shape[0]
    
    fov = 3.1
    
        
    # Homogenize points
    ones = torch.ones((N, 1), device=device)
    world_points = torch.cat([world_points, ones], dim=-1)
    cam_space = (model_view @ world_points.T).T
    
    # Perspective projection
    f = 1 / torch.tan(torch.tensor(fov, device=device) / 2)
    far = 100.0
    near = 0.0
    
    
    
    if cam_space.shape[0] == 0:
        return torch.empty((0, 2), device=device)
    
    projection_matrix = torch.zeros((4, 4), device=device)
    projection_matrix[0, 0] = - 0.16 * f / aspect
    projection_matrix[1, 1] = 0.9 * f
    projection_matrix[2, 2] = -(far + near) / (far - near)
    projection_matrix[2, 3] = -(2 * far * near) / (far - near)
    projection_matrix[3, 2] = -1
    projection_matrix[3, 3] = 0
    
    proj_space = (projection_matrix @ cam_space.T).T
    
    screen_x = (proj_space[:, 0] + 1) / 2 * SCREEN_WIDTH
    screen_y = (1 - proj_space[:, 1]) / 2 * SCREEN_HEIGHT
    
    
    
    
    
    
    return torch.stack([screen_x, screen_y], dim=-1)
