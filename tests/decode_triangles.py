import torch

def decode_kcl_triangles(pos, fnrm, enrm1, enrm2, enrm3, height):
    """
    Decode KCL prism data into triangle vertices.

    Args:
        pos   : (N, 3) tensor of base positions
        fnrm  : (N, 3) tensor of face normals
        enrm1 : (N, 3) tensor of edge normal 1
        enrm2 : (N, 3) tensor of edge normal 2
        enrm3 : (N, 3) tensor of edge normal 3
        height: (N,)   tensor of heights

    Returns:
        triangles: (N, 3, 3) tensor of vertices per prism triangle
    """
    
    enrm1 = enrm1 / torch.norm(enrm1, dim=-1, keepdim=True)
    enrm2 = enrm2 / torch.norm(enrm2, dim=-1, keepdim=True)
    enrm3 = enrm3 / torch.norm(enrm3, dim=-1, keepdim=True)
    fnrm = fnrm / torch.norm(fnrm, dim=-1, keepdim=True)
    
    # Cross products
    cross_a = torch.cross(enrm1, fnrm, dim=-1)  # (N, 3)
    cross_b = torch.cross(enrm2, fnrm, dim=-1)  # (N, 3)

    # Dot products for scaling
    dot_a = torch.sum(cross_a * enrm3, dim=-1)  # (N,)
    dot_b = torch.sum(cross_b * enrm3, dim=-1)  # (N,)

    # Avoid divide-by-zero
    dot_a = torch.where(dot_a == 0, torch.ones_like(dot_a), dot_a)
    dot_b = torch.where(dot_b == 0, torch.ones_like(dot_b), dot_b)

    # Scale factors
    scale_a = (height / dot_a)[..., None]  # (N, 1)
    scale_b = (height / dot_b)[..., None]  # (N, 1)

    # Vertices
    v1 = pos
    v2 = pos + cross_b * scale_b
    v3 = pos + cross_a * scale_a

    return torch.stack([v1, v2, v3], dim=1)  # (N, 3, 3)


if __name__ == "__main__":
    # Single example, shaped as (1, 3)
    pos   = torch.tensor([[-1750.0, 600.7471, -1450.0]])
    fnrm  = torch.tensor([[9.0723e-01, -1.8556e+03, 4.0310e+03]])
    enrm1 = torch.tensor([[4.2090e-01, -1.6809e+03, 3.6650e+03]])
    enrm2 = torch.tensor([[-65407.9766, 1871.9451, -4015.0022]])
    enrm3 = torch.tensor([[15.579, -44671.0, 47952.0]])
    height = torch.tensor([36.9568])
    
    triangles = decode_kcl_triangles(pos, fnrm, enrm1, enrm2, enrm3, height)
    print(triangles)