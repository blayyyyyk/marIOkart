from .utils import (
    read_u16,
    read_vector_3d_fx16,
    read_vector_3d_fx32,
    read_u32,
    read_fx32,
    read_s32,
    slice_bits
)
from typing import Sequence, Self


class PrismsBase:
    """
    Represents the triangular prisms section of the KCL file.

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
    def __init__(self,
        _height: Sequence[float],
        _pos_i: Sequence[int],
        _fnrm_i: Sequence[int],
        _enrm1_i: Sequence[int],
        _enrm2_i: Sequence[int],
        _enrm3_i: Sequence[int],
        _attributes: Sequence[Sequence[int]]
    ):
        self.height = _height
        self.pos_i = _pos_i
        self.fnrm_i = _fnrm_i
        self.enrm1_i = _enrm1_i
        self.enrm2_i = _enrm2_i
        self.enrm3_i = _enrm3_i
        self.attributes = _attributes
        
    @classmethod
    def from_bytes(cls, data: bytes) -> Self:
        iter = range(0, len(data), 0x10)
        return cls(
            [read_fx32(data, i + 0x00) for i in iter],
            [read_u16(data, i + 0x04) for i in iter],
            [read_u16(data, i + 0x06) for i in iter],
            [read_u16(data, i + 0x08) for i in iter],
            [read_u16(data, i + 0x0A) for i in iter],
            [read_u16(data, i + 0x0C) for i in iter],
            [PrismsBase.parse_attributes(read_u16(data, i + 0x0E)) for i in iter]
        )
        
    @staticmethod
    def parse_attributes(bits: int) -> list[int]:
        return [
            slice_bits(bits, 0, 1), # map 2d shadow
            slice_bits(bits, 1, 4), # light id (bit 1-3)
            slice_bits(bits, 4, 5), # ignore drivers
            slice_bits(bits, 5, 8), # collision variant
            slice_bits(bits, 8, 13), # collision type
            slice_bits(bits, 13, 14), # ignore items
            slice_bits(bits, 14, 15), # is wall
            slice_bits(bits, 15, 16) # is floor
        ]


class KCLBase:
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
    prism_cls = PrismsBase
    
    def __init__(self, 
        data: bytes,
        prisms: PrismsBase,
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
        
    ):
        self.data = data
        self.prisms = prisms
        self.positions = positions
        self.normals = normals
        self.positions_offset = _positions_offset
        self.normals_offset = _normals_offset
        self.prisms_offset = _prisms_offset
        self.block_data_offset = _block_data_offset
        self.prism_thickness = _prism_thickness
        self.area_min_pos = _area_min_pos
        self.area_x_width_mask = _area_x_width_mask
        self.area_y_width_mask = _area_y_width_mask
        self.area_z_width_mask = _area_z_width_mask
        self.block_width_shift = _block_width_shift
        self.area_x_blocks_shift = _area_x_blocks_shift
        self.area_xy_blocks_shift = _area_xy_blocks_shift
        self.sphere_radius = _sphere_radius
        
    @classmethod
    def from_bytes(cls, data: bytes, **kwargs) -> Self:
        _positions_offset = read_u32(data, 0x00)
        _normals_offset = read_u32(data, 0x04)
        _prisms_offset = read_u32(data, 0x08)
        _block_data_offset = read_u32(data, 0x0C)
        _prism_thickness = read_fx32(data, 0x10)
        _area_min_pos = read_vector_3d_fx32(data, 0x14)
        _area_x_width_mask = read_u32(data, 0x20)
        _area_y_width_mask = read_u32(data, 0x24)
        _area_z_width_mask = read_u32(data, 0x28)
        _block_width_shift = read_u32(data, 0x2C)
        _area_x_blocks_shift = read_u32(data, 0x30)
        _area_xy_blocks_shift = read_u32(data, 0x34)
        _sphere_radius = None#read_f32(data, 0x38)
        prisms = PrismsBase.from_bytes(data[_prisms_offset+0x10:_block_data_offset])
        positions = KCLBase._parse_positions(data, prisms, _positions_offset)
        normals = KCLBase._parse_normals(data, prisms, _normals_offset)
        prisms = cls.prism_cls(
            prisms.height,
            prisms.pos_i,
            prisms.fnrm_i,
            prisms.enrm1_i,
            prisms.enrm2_i,
            prisms.enrm3_i,
            prisms.attributes,
            **kwargs
        )
        return cls(
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
            **kwargs
        )
        
        
        
    @staticmethod
    def _parse_positions(data: bytes, prisms: PrismsBase, positions_offset: int):
        """
        Parse position vectors from the file.

        Each position vector consists of 3 consecutive floats (X, Y, Z).
        The number of positions is determined from the prisms indices.

        Returns
        -------
        list
            List of 3D vectors (tuples of floats)
        """
        position_size = 0x0C
        start = positions_offset
        section_size = max(prisms.pos_i)
        end = (section_size + 1) * position_size + start
        positions = []
        for offset in range(start, end, position_size):
            positions.append(read_vector_3d_fx32(data, offset))
            
        return positions

    @staticmethod
    def _parse_normals(data: bytes, prisms: PrismsBase, normals_offset: int):
        """
        Parse normal vectors from the file.

        Each normal vector consists of 3 consecutive floats (X, Y, Z).
        The number of normals is determined from all prism normal indices.

        Returns
        -------
        list
            List of 3D normal vectors (tuples of floats)
        """
        normal_size = 0x06
        start = normals_offset
        section_size = max([
            *prisms.fnrm_i,
            *prisms.enrm1_i,
            *prisms.enrm2_i,
            *prisms.enrm3_i,
        ])
        end = (section_size + 1) * normal_size + start
        normals = []
        for offset in range(start, end, normal_size):
            normals.append(read_vector_3d_fx16(data, offset))
            
        return normals
         
    def search_block(self, point: tuple[float, float, float] | list[float, float, float]):
        """
        Return the offset of the leaf node containing a queried point
        """
        block_start = self.block_data_offset
        block_data = self.data[block_start:]
        
        px, py, pz = point
        minx, miny, minz = self.area_min_pos

        x = int(px - minx)
        if (x & self.area_x_width_mask) != 0:
            return None

        y = int(py - miny)
        if (y & self.area_y_width_mask) != 0:
            return None

        z = int(pz - minz)
        if (z & self.area_z_width_mask) != 0:
            return None

        # initialize root
        shift = self.block_width_shift
        cur_block_offset = 0  # root at start of block_data

        index = 4 * (((z >> shift) << self.area_xy_blocks_shift)
            | ((y >> shift) << self.area_x_blocks_shift)
                    | (x >> shift))

        while True:
            offset = read_u32(block_data, cur_block_offset + index)

            if (offset & 0x80000000) != 0:
                # negative flag = leaf node
                break

            shift -= 1
            cur_block_offset += offset

            # initialize next index
            index = 4 * (((z >> shift) & 1) << 2
                        | ((y >> shift) & 1) << 1
                        | ((x >> shift) & 1))

        # leaf = return pointer into block_data (as slice)
        leaf_offset = cur_block_offset + (offset & 0x7FFFFFFF)
        return leaf_offset
        
        
        
class Prisms(PrismsBase):
    """
    Represents the triangular prisms section of the KCL file.

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

    def __init__(self, 
        _height: list[float],
        _pos_i: list[int],
        _fnrm_i: list[int],
        _enrm1_i: list[int],
        _enrm2_i: list[int],
        _enrm3_i: list[int],
        _attributes: list[list[int]]
    ):
        super().__init__(
            _height,
            _pos_i,
            _fnrm_i,
            _enrm1_i,
            _enrm2_i,
            _enrm3_i,
            _attributes
        )
        self.height = _height
        self.pos_i = _pos_i
        self.fnrm_i = _fnrm_i
        self.enrm1_i = _enrm1_i
        self.enrm2_i = _enrm2_i
        self.enrm3_i = _enrm3_i
        self.attributes = _attributes

    def __getitem__(self, idx):
        """
        Return all attributes of the prism at index ``idx``.
        """
        if idx >= len(self):
            raise IndexError("Index out of range")
        return [arr[idx] for arr in self.__dict__.values()]

    def __len__(self):
        """
        Return the number of prisms in the section.
        """
        return len(self.pos_i)

    def __iter__(self):
        """
        Iterate over all prisms, yielding full attribute lists.
        """
        for i in range(len(self)):
            yield self[i // 0x10]    
        
        

class KCL(KCLBase):
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
    prism_cls = Prisms
    
    def __init__(
        self, 
        data: bytes,
        prisms: Prisms,
        positions: list[list[float]],
        normals: list[list[float]],
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
    ):
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
            _sphere_radius
        )
        self.data = data
        self.prisms = prisms
        self.positions = positions
        self.normals = normals
        self.positions_offset = _positions_offset
        self.normals_offset = _normals_offset
        self.prisms_offset = _prisms_offset
        self.block_data_offset = _block_data_offset
        self.prism_thickness = _prism_thickness
        self.area_min_pos = _area_min_pos
        self.area_x_width_mask = _area_x_width_mask
        self.area_y_width_mask = _area_y_width_mask
        self.area_z_width_mask = _area_z_width_mask
        self.block_width_shift = _block_width_shift
        self.area_x_blocks_shift = _area_x_blocks_shift
        self.area_xy_blocks_shift = _area_xy_blocks_shift
        self.sphere_radius = _sphere_radius

    def __str__(self):
        """
        Returns a human-readable summary of the KCL file.

        Output includes:
        - Number of positions and a preview of first and last vectors
        - Number of normals and a preview of first and last vectors
        - Number of prisms
        """
        def str_vec(l):
            return f"""
            {l[0]}
            ...
            {len(l)} vectors
            ...
            {l[-1]}
            """

        return f"""
        Positions:
        {str_vec(self.positions) if len(self.positions) != 0 else "No entries"}
        Normals:
        {str_vec(self.normals) if len(self.normals) != 0 else "No entries"}
        Prisms:
        {len(self.prisms) if len(self.prisms) != 0 else "No entries"}\n
        """
        
    
 
    @classmethod
    def from_file(cls, path: str):
        data = None    
        with open(path, 'rb') as f:
            data = f.read()
            
        assert data is not None
        return cls.from_bytes(data)