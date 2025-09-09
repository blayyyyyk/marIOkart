import torch
import torch.nn.functional as F

def get_mps_device() -> torch.device:
    # Check that MPS is available
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                  "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                  "and/or you do not have an MPS-enabled device on this machine.")
    
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
    
    valid_mask = ((u >= 0) & (u <= 1) & (t >= 0))
    
    return t[valid_mask], u[valid_mask]
    
def pairwise_distances(pts):
    norms = (pts**2).sum(dim=-1)
    G = pts @ pts.T
    D2 = norms[:, None] + norms[None, :] - 2 * G
    D2 = torch.clamp(D2, min=0.0)
    D = torch.sqrt(D2)
    return D