from __future__ import annotations
from mkds.kcl import KCLBase, PrismsBase
from mkds.utils import read_u16, read_u32
from typing import Sequence
import torch


class PrismsTensor(PrismsBase):
    """
    Represents the triangular prisms section of the KCL file.

    Each prism is a 0x10 byte structure with the following layout:

    .. include:: /_includes/kcl_tables.rst
      :start-after: .. _kcl-table-prisms:
      :end-before: .. _kcl-table:

    Attributes
    ----------
    _height : list[float]
        Prism heights
    _pos_i : list[int]
        Vertex indices
    _fnrm_i : list[int]
        Face normal indices
    _enrm1_i : list[int]
        Edge normal 1 indices
    _enrm2_i : list[int]
        Edge normal 2 indices
    _enrm3_i : list[int]
        Edge normal 3 indices
    _attributes : list[int]
        Collision attribute flags
    """

    def __init__(
        self,
        _height: Sequence[float],
        _pos_i: Sequence[int],
        _fnrm_i: Sequence[int],
        _enrm1_i: Sequence[int],
        _enrm2_i: Sequence[int],
        _enrm3_i: Sequence[int],
        _attributes: Sequence[Sequence[int]],
        device=None,
    ):
        super().__init__(
            _height, _pos_i, _fnrm_i, _enrm1_i, _enrm2_i, _enrm3_i, _attributes
        )
        self.height = torch.tensor(_height, dtype=torch.float32, device=device)
        self.pos_i = torch.tensor(_pos_i, dtype=torch.int32, device=device)
        self.fnrm_i = torch.tensor(_fnrm_i, dtype=torch.int32, device=device)
        self.enrm1_i = torch.tensor(_enrm1_i, dtype=torch.int32, device=device)
        self.enrm2_i = torch.tensor(_enrm2_i, dtype=torch.int32, device=device)
        self.enrm3_i = torch.tensor(_enrm3_i, dtype=torch.int32, device=device)
        self.attributes = torch.tensor(_attributes, dtype=torch.int32, device=device)

    @property
    def map_2d_shadow(self):
        return self.attributes[:, 0]

    @property
    def light_id(self):
        return self.attributes[:, 1]

    @property
    def ignore_drivers(self):
        return self.attributes[:, 2]

    @property
    def collision_variant(self):
        return self.attributes[:, 3]

    @property
    def collision_type(self):
        return self.attributes[:, 4]

    @property
    def ignore_items(self):
        return self.attributes[:, 5]

    @property
    def is_wall(self):
        return self.attributes[:, 6]

    @property
    def is_floor(self):
        return self.attributes[:, 7]


class KCLTensor(KCLBase):
    """
    Represents a KCL (collision) file.

    KCL files store simplified model data for collision detection in games
    such as Mario Kart Wii / DS. They consist of a header, positions, normals,
    triangular prisms, and octree blocks.

    .. include:: /_includes/kcl_tables.rst
      :start-after: .. _kcl-table:
      :end-before: .. _kcl-end:

    Attributes
    ----------
    _positions_offset : int
        File offset to position vectors
    _normals_offset : int
        File offset to normal vectors
    _prisms_offset : int
        File offset to prism data
    _block_data_offset : int
        File offset to octree blocks
    _prism_thickness : float
        Depth of each prism
    _area_min_pos : list[float]
        Minimum coordinates of the collision area
    _area_x_width_mask : int
        X-axis mask for octree
    _area_y_width_mask : int
        Y-axis mask for octree
    _area_z_width_mask : int
        Z-axis mask for octree
    _block_width_shift : int
        Octree leaf size shift
    _area_x_blocks_shift : int
        Root block child index shift (Y)
    _area_xy_blocks_shift : int
        Root block child index shift (Z)
    _sphere_radius : float or None
        Optional maximum sphere radius for collisions
    _prisms : Prisms
        Parsed prism objects
    _positions : list
        List of vertex positions
    _normals : list
        List of normal vectors
    """

    prism_cls = PrismsTensor

    def __init__(
        self,
        data: bytes,
        prisms: PrismsTensor,
        positions: Sequence[Sequence[float]],
        normals: Sequence[Sequence[float]],
        _positions_offset: int,
        _normals_offset: int,
        _prisms_offset: int,
        _block_data_offset: int,
        _prism_thickness: float,
        _area_min_pos: tuple[float, float, float],
        _area_x_width_mask: int,
        _area_y_width_mask: int,
        _area_z_width_mask: int,
        _block_width_shift: int,
        _area_x_blocks_shift: int,
        _area_xy_blocks_shift: int,
        _sphere_radius: int | None,
        device=None,
    ):
        _positions = torch.tensor(positions, device=device)
        _normals = torch.tensor(normals, device=device)
        super().__init__(
            data,
            prisms,
            positions,
            normals,
            _positions_offset,
            _normals_offset,
            _prisms_offset,
            _block_data_offset,
            _prism_thickness,
            _area_min_pos,
            _area_x_width_mask,
            _area_y_width_mask,
            _area_z_width_mask,
            _block_width_shift,
            _area_x_blocks_shift,
            _area_xy_blocks_shift,
            _sphere_radius,
        )
        self.prisms = prisms
        self.positions = _positions
        self.normals = _normals
        self.device = device
        self.triangles = self._compute_triangles()

    def _compute_triangles(self):
        # Indexed Vectors
        height = self.prisms.height
        vertex_0 = self.positions[self.prisms.pos_i]
        face_norm = self.normals[self.prisms.fnrm_i]
        edge_norm_0 = self.normals[self.prisms.enrm1_i]
        edge_norm_1 = self.normals[self.prisms.enrm2_i]
        edge_norm_2 = self.normals[self.prisms.enrm3_i]

        cross_a = edge_norm_0.cross(face_norm, dim=-1)
        cross_b = edge_norm_1.cross(face_norm, dim=-1)

        vertex_1 = (
            vertex_0
            + cross_b * (height / torch.linalg.vecdot(edge_norm_2, cross_b))[:, None]
        )
        vertex_2 = (
            vertex_0
            + cross_a * (height / torch.linalg.vecdot(edge_norm_2, cross_a))[:, None]
        )

        out = torch.stack([vertex_0, vertex_1, vertex_2], dim=1)

        return out

    def search_triangles(
        self,
        point: tuple[float, float, float] | torch.Tensor,
        filter_attribute_id: int | None = None,
    ):
        assert self.triangles is not None
        if not isinstance(point, tuple):
            p = tuple(point.tolist())
            leaf_offset = self.search_block(p)
        else:
            leaf_offset = self.search_block(point)

        if leaf_offset is None:
            return None

        tri_indices: list[int] = []
        chunk_size = 0x02
        start = self.block_data_offset + leaf_offset + chunk_size
        for data_offset in range(start, len(self.data), chunk_size):
            idx = read_u16(self.data, data_offset) - 1
            if idx == -1:
                break

            tri_indices.append(idx)

        if len(tri_indices) == 0:
            return None

        return tri_indices

    def nearest_triangles(self, point, n=1, device=None):
        tri_indices = self.search_triangles(point)
        assert tri_indices is not None

        tri_vertices = self.triangles[tri_indices]  # (M, 3, 3)
        point_t = torch.tensor(point, dtype=torch.float32, device=device)
        dists = KCLTensor._point_triangle_distance_squared(
            point_t, tri_vertices, device=device
        )

        nearest_idx = torch.topk(dists, k=min(n, len(dists)), largest=False).indices
        return tri_vertices[nearest_idx], tri_indices[nearest_idx]

    @staticmethod
    def _point_triangle_distance_squared(point, triangles: torch.Tensor, device=None):
        """
        Vectorized squared distance from a point to many triangles.
        point: (3,) tensor
        triangles: (M, 3, 3) tensor
        Returns: (M,) distances squared
        """
        p = point.unsqueeze(0).unsqueeze(1)  # (1,1,3)
        a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]  # (M,3)

        ab = b - a
        ac = c - a
        ap = p - a.unsqueeze(1)

        d1 = (ab * ap).sum(-1)
        d2 = (ac * ap).sum(-1)
        mask = (d1 <= 0) & (d2 <= 0)
        dist_a = (ap**2).sum(-1)

        bp = p - b.unsqueeze(1)
        d3 = (ab * bp).sum(-1)
        d4 = (ac * bp).sum(-1)
        mask_b = (d3 >= 0) & (d4 <= d3)
        dist_b = (bp**2).sum(-1)

        cp = p - c.unsqueeze(1)
        d5 = (ab * cp).sum(-1)
        d6 = (ac * cp).sum(-1)
        mask_c = (d6 >= 0) & (d5 <= d6)
        dist_c = (cp**2).sum(-1)

        # Default large value
        dist = torch.full((triangles.shape[0],), float("inf"), device=device)

        dist[mask] = dist_a[mask]
        dist[mask_b] = dist_b[mask_b]
        dist[mask_c] = dist_c[mask_c]

        # Fallback for points projecting inside face
        inside_mask = ~(mask | mask_b | mask_c)
        if inside_mask.any():
            normal = torch.cross(ab, ac, dim=-1)
            normal = normal / normal.norm(dim=-1, keepdim=True)
            dist_face = ((ap.squeeze(1) * normal).sum(-1) ** 2)[inside_mask]
            dist[inside_mask] = dist_face

        return dist

    @classmethod
    def from_file(cls, path: str, device=None):
        data = None
        with open(path, "rb") as f:
            data = f.read()

        assert data is not None
        return cls.from_bytes(data, device=device)
