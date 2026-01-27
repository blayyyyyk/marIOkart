import ctypes, math, json, torch, numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
from numpy.typing import NDArray
from desmume.emulator import (
    DeSmuME,
    DeSmuME_Memory,
    SCREEN_WIDTH,
    SCREEN_HEIGHT
)
from typing import (
    Type,
    TypeVar,
    cast,
    Union,
    TypedDict,
    Literal,
    Generic,
    cast,
    Optional
)
from typing import Annotated, Literal
from typing_extensions import override
from private.mkdslib import (
    VecFx32,
    VecFx16,
    camera_t,
    driver_t,
    race_state_t,
    race_status_t,
    mdat_mapdata_t,
    kcol_header_t,
    struct_kcol_prism_data_t,
    struct_nkm_cpoi_entry_t,
    struct_race_driver_status_t,
    RACER_PTR_ADDR,
    CAMERA_PTR_ADDR,
    RACE_STATE_PTR_ADDR,
    RACE_STATUS_PTR_ADDR,
    MAP_DATA_PTR_ADDR,
    COLLISION_DATA_ADDR,
    FX32_SCALE_FACTOR
)
from src.utils.vector import *
from io import BytesIO

NUM_RAYS = 16
MAX_DIST = 3000.0

class OctreeNode(ctypes.LittleEndianStructure):
    _fields_ = [
        # The lowest 31 bits are the offset
        ("offset", ctypes.c_uint32, 31),
        # The highest bit (31) is the leaf flag
        ("is_leaf", ctypes.c_uint32, 1)
    ]

class MMUPrefix(ctypes.Structure):
    _fields_ = [
        ("ARM9_ITCM", ctypes.c_uint8 * 0x8000),
        ("ARM9_DTCM", ctypes.c_uint8 * 0x4000),
        ("MAIN_MEM", ctypes.c_uint8 * (16 * 1024 * 1024)),
        # ...
    ]


T = TypeVar('T', bound=Union[ctypes.Structure, ctypes.Array])
H = TypeVar('H', bound=ctypes.Structure)

_PrismEntriesAttributes = TypedDict('_PrismEntriesAttributes', {
    'shadow_2d': np.ndarray,     
    'light_id': np.ndarray,      
    'ignore_drivers': np.ndarray,
    'variant': np.ndarray,       
    'collision_type': np.ndarray,
    'ignore_items': np.ndarray,  
    'is_wall': np.ndarray,       
    'is_floor': np.ndarray,      
})
def _unpack_col_attributes(raw_attrs: np.ndarray) -> _PrismEntriesAttributes:
    """
    Vectorized version of parse_attributes.

    Input:  (N,) array of uint16 (or int32)
    Output: (N,) structured array with named fields
    """
    # Define the output format
    parsed_dtype = np.dtype([
        ('shadow_2d',     'u1'), # bit 0
        ('light_id',      'u1'), # bits 1-4 (3 bits)
        ('ignore_drivers','u1'), # bit 4
        ('variant',       'u1'), # bits 5-8 (3 bits)
        ('collision_type','u1'), # bits 8-13 (5 bits)
        ('ignore_items',  'u1'), # bit 13
        ('is_wall',       'u1'), # bit 14
        ('is_floor',      'u1'), # bit 15
    ])
    

    result = np.zeros(raw_attrs.shape, dtype=parsed_dtype)

    # Apply Bitwise Logic (Vectorized)
    result['shadow_2d']      = (raw_attrs >> 0) & 0x1
    result['light_id']       = (raw_attrs >> 1) & 0x7
    result['ignore_drivers'] = (raw_attrs >> 4) & 0x1
    result['variant']        = (raw_attrs >> 5) & 0x7
    result['collision_type'] = (raw_attrs >> 8) & 0x1F
    result['ignore_items']   = (raw_attrs >> 13) & 0x1
    result['is_wall']        = (raw_attrs >> 14) & 0x1
    result['is_floor']       = (raw_attrs >> 15) & 0x1

    return cast(_PrismEntriesAttributes, result)

def _pack_i4_fx32(arr):
    repacked = structured_to_unstructured(arr, dtype=np.float32)
    return repacked / (1 << 12)

def _norm(arr, epsilon=1e-8):
    n = np.linalg.norm(arr, axis=-1, keepdims=True)
    n[n < epsilon] = epsilon
    return arr / n

vmap_over_triangles = torch.vmap(
    ray_triangle_intersection,
    in_dims=(0, None, None, None, None)
)

vmap_all_pairs = torch.vmap(
    vmap_over_triangles,
    in_dims=(None, 0, 0, None, None)
)

_PrismEntries = TypedDict('_PrismEntries', {
    "height": NDArray[np.int32], # fx32
    "posIdx": NDArray[np.int32],
    "fNrmIdx": NDArray[np.int32],
    "eNrm1Idx": NDArray[np.int32],
    "eNrm2Idx": NDArray[np.int32],
    "eNrm3Idx": NDArray[np.int32],
    "attribute": NDArray[np.int16],
})
_CollisionEntries = TypedDict('_CollisionEntries', {
    'prism': _PrismEntries,
    'prism_attribute': _PrismEntriesAttributes,
    'vert': Annotated[NDArray[np.float32], Literal["N", 3]],
    'nrm': Annotated[NDArray[np.float32], Literal["N", 3]]
})
_TriangleVertices = TypedDict('_TriangleVertices', {
    'v1': torch.Tensor,
    'v2': torch.Tensor,
    'v3': torch.Tensor 
})
_CheckpointPos = TypedDict('_CheckpointPos', {
    'current_checkpoint_id': int,
    'current_checkpoint_pos': torch.Tensor,
    'next_checkpoint_id': int,
    'next_checkpoint_pos': torch.Tensor,
})
_CheckpointAngle = TypedDict('_CheckpointAngle', {
    'midpoint_angle': torch.Tensor
})
RayCastInfo = TypedDict('RayCastInfo', {'distance': torch.Tensor, 'position': torch.Tensor, 'mask': torch.Tensor})
class CheckpointInfo(_CheckpointPos, _CheckpointAngle): pass
class CollisionData(_CollisionEntries, _TriangleVertices): pass


class MarioKart_Memory(DeSmuME_Memory):
    """
    A memory interface for Mario Kart DS emulator data access. Extends DeSmuME_Memory to read and process
    game data structures including driver information, camera settings, race status, collision data, and
    checkpoint information. Provides utilities for memory reading, collision detection, coordinate space
    transformations, and ray casting for obstacle detection.
    
    Attributes
    ----------
    `_memoryview` (`memoryview`): Direct access to emulator main memory.\n
    `_driver` (`driver_t`): Cached driver structure.\n
    `_camera` (`camera_t`): Cached camera structure.\n
    `_race_status` (`race_status_t`): Cached race status structure.\n
    `_map_data` (`mdat_mapdata_t`): Cached map data structure.\n
    `_cpoi_data` (`ctypes`.Array): Cached checkpoint data array.\n
    `_kcl_header` (`kcol_header_t`): Cached collision header.\n
    `_kcl_data` (`dict`): Cached processed collision data.\n
    `_race_state` (`race_state_t`): Cached race state structure.\n
    `_ready` (`bool`): Flag indicating if race has started.\n
    `_device` (`torch`.device): Target device for tensor operations.
    
    Methods
    -------
    `read_struct`: Read a ctypes structure from memory at a given address.\n
    `memoryview`: Property for accessing main memory view.\n
    `driver`: Property for cached driver data.\n
    `camera`: Property for cached camera data.\n
    `race_status`: Property for cached race status.\n
    `map_data`: Property for cached map data.\n
    `checkpoint_data`: Property for cached checkpoint array.\n
    `collision_header`: Property for cached collision header.\n
    `race_state`: Property for cached race state.\n
    `race_ready`: Property indicating if race has started.\n
    `collision_data`: Property for processed collision triangles and attributes.\n
    `driver_status`: Property for current driver status.\n
    `frame_count_race_start`: Property for frame count at race start.\n
    `frame_count`: Property for current frame count.\n
    `get_camera_settings`: Retrieve camera FOV, aspect ratio, and clipping planes.\n
    `project_to_screen`: Transform 3D world points to 2D screen space with depth.\n
    `checkpoint_pos`: Get current and next checkpoint positions.\n
    `checkpoint_angle`: Calculate checkpoint midpoint angle in driver local space.\n
    `checkpoint_info`: Combined checkpoint position and angle information.\n
    `read_facing_point_checkpoint`: Calculate intersection of driver direction ray with next checkpoint line.\n
    `obstacle_info`: Perform ray casting to detect nearby obstacles and walls.\n
    `get_obs`: Aggregate obstacle distances and checkpoint angle for observation.\n
    `collision_search`: Find collision triangles at a given 3D point using octree search.\n
    `set_torch_device`: Set the target device for tensor operations.\n
    `get_torch_device`: Retrieve the current target device.
    """
    def __init__(self, *args, device: torch.device, **kwargs) -> None:
        """
        Initialize the emulator with memory access and state management.
        
        Args
        ----
        *args: Variable length argument list passed to parent class.
            device (torch.device): The torch device (CPU or GPU) to use for computations.
            **kwargs: Arbitrary keyword arguments passed to parent class.
        
        Raises:
            AssertionError: If the emulator library is not loaded (self.emu.lib is None).
        
        Attributes:
            _memoryview: Memory view of the emulator's main memory.
            _driver: Optional driver state structure.
            _camera: Optional camera state structure.
            _race_status: Optional race status structure.
            _map_data: Optional map data structure.
            _cpoi_data: Optional collision point of interest data array.
            _kcl_header: Optional KCL collision header structure.
            _kcl_data: Optional processed collision data.
            _race_state: Optional race state structure.
            _ready: Flag indicating if emulator is ready (initialized as False).
            _device: The torch device for computations.
        """
        super().__init__(*args, **kwargs)
        assert self.emu.lib is not None
        self._memoryview = memoryview(MMUPrefix.in_dll(self.emu.lib, "MMU").MAIN_MEM)
        self._driver: Optional[driver_t] = None
        self._camera: Optional[camera_t] = None
        self._race_status: Optional[race_status_t] = None
        self._map_data: Optional[mdat_mapdata_t] = None
        self._cpoi_data: Optional[ctypes.Array[struct_nkm_cpoi_entry_t]] = None
        self._kcl_header: Optional[kcol_header_t] = None
        self._kcl_data: Optional[CollisionData] = None
        self._race_state: Optional[race_state_t] = None
        self._ready = False
        self._device = device

    def read_struct(self, struct_t: Type[T], addr: int) -> T:
        """
        Read a structured data type from memory at the specified address.
        
        Args
        ----
        struct_t (Type[T]): The ctypes.Structure type to read from memory.
            addr (int): The memory address to read from.
        
        Returns
        -------
        `T`: An instance of the specified `ctypes.Structure` type populated with data from memory.
        """
        return MarioKart_Memory._read_struct(self.memoryview, struct_t, addr) # main memory region

    @property
    def memoryview(self) -> memoryview:
        """
        Get a memoryview object of the emulator's memory.

        Returns
        -------
        `memoryview`: A memoryview object providing a buffer interface to the emulator's memory.
        """
        return self._memoryview

    @staticmethod
    def _read_struct(mem, struct_t: Type[T], addr: int) -> T:
        struct = struct_t.from_buffer(mem, addr - 0x02000000) # main memory region
        return struct

    @property
    def driver(self) -> driver_t:
        """
        Get the driver object for the current racer.
        
        Lazily loads the driver data from memory on first access by reading from the
        racer pointer address and parsing it as a driver_t struct. Subsequent calls
        return the cached driver object.
        
        Returns
        -------
        `driver_t`: The driver object associated with the current racer.
        """
        if self._driver is None:
            driver_addr = self.unsigned.read_long(RACER_PTR_ADDR)
            self._driver = self.read_struct(driver_t, driver_addr)

        return self._driver
    
    @property
    def camera(self) -> camera_t:
        """
        Retrieve the camera structure from emulator memory.
        
        Lazily loads and caches the camera data from the emulator's memory.
        On first access, reads the camera pointer address and constructs the
        camera structure. Subsequent calls return the cached camera object.
        
        Returns
        -------
        `camera_t`: The camera structure containing camera state and properties.
        """
        if self._camera is None:
            camera_addr = self.unsigned.read_long(CAMERA_PTR_ADDR)
            self._camera = self.read_struct(camera_t, camera_addr)

        return self._camera

    @property
    def race_status(self) -> race_status_t:
        """
        Retrieve the current race status information.

        Returns
        -------
        `race_status_t`: A structured object containing race status data including
                           track information, player position, lap count, and race state.
                           The result is cached after the first read to avoid repeated
                           memory access.

        Raises:
        `Exception`: May raise exceptions from read_struct or memory read operations
                       if the race status pointer address is invalid.
        """
        if self._race_status is None:
            race_status_addr = self.unsigned.read_long(RACE_STATUS_PTR_ADDR)
            self._race_status = self.read_struct(race_status_t, race_status_addr)

        return self._race_status

    @property
    def map_data(self) -> mdat_mapdata_t:
        """
        Retrieve the map data structure from emulator memory.
        
        Lazily loads and caches the map data by reading from the MAP_DATA_PTR_ADDR
        address. Subsequent calls return the cached value without re-reading memory.
        
        Returns
        -------
        `mdat_mapdata_t`: The map data structure containing map information.
        """
        if self._map_data is None:
            map_data_addr = self.unsigned.read_long(MAP_DATA_PTR_ADDR)
            self._map_data = self.read_struct(mdat_mapdata_t, map_data_addr)

        return self._map_data

    @property
    def checkpoint_data(self) -> ctypes.Array[struct_nkm_cpoi_entry_t]:
        """
        Retrieve the checkpoint data from the emulator's map.
        
        Lazily loads and caches the checkpoint point of interest (cpoi) data structure
        from the emulator's memory. On first call, reads the cpoi array from memory based
        on the map data's cpoi count and offset. Subsequent calls return the cached data.
        
        Returns
        -------
        `ctypes.Array[struct_nkm_cpoi_entry_t]`: A ctypes array of checkpoint entries
                containing checkpoint data for the current map. The array length corresponds
                to the map's checkpoint count.
        """
        if self._cpoi_data is None:
            mdata = self.map_data
            CpoiArrayType = struct_nkm_cpoi_entry_t * int(mdata.cpoiCount)
            self._cpoi_data = self.read_struct(CpoiArrayType, mdata.cpoi.value)

        return self._cpoi_data

    @property
    def collision_header(self) -> kcol_header_t:
        """
        Retrieve the collision header structure from the emulator.
        
        Lazily loads and caches the collision header data from the emulator's memory
        at the predefined COLLISION_DATA_ADDR address. Subsequent calls return the
        cached header without re-reading from memory.
        
        Returns
        -------
        `kcol_header_t`: The collision header structure containing collision data metadata.
        """
        if self._kcl_header is None:
            self._kcl_header = self.read_struct(kcol_header_t, COLLISION_DATA_ADDR)

        return self._kcl_header

    @property
    def race_state(self) -> race_state_t:
        """
        Retrieve the current race state.

        Returns the cached race state object if it has been previously loaded.
        Otherwise, reads the race state from memory at the address stored in
        RACE_STATE_PTR_ADDR, caches it, and returns it.

        Returns
        -------
        `race_state_t`: The current race state object containing race information.
        """
        if self._race_state is None:
            addr = self.unsigned.read_long(RACE_STATE_PTR_ADDR)
            self._race_state = self.read_struct(race_state_t, addr)

        return self._race_state

    @property
    def race_ready(self):
        """
        Determine if the race is ready to start based on frame counter state.
        
        Checks if the difference between the current frame counter and a secondary
        frame counter equals 1, indicating the race is in a ready state. Once ready
        is set to True, it remains True for subsequent calls.
        
        Returns
        -------
        `bool`: True if the race is ready, False otherwise.
        """
        if not self._ready:
            try:
                _f = self.race_state.frameCounter - self.race_state.frameCounter2
                self._ready = _f == 1
            except:
                pass

        return self._ready

    X = TypeVar('X', bound=ctypes.Structure)
    def _np_entries(self, c_struct: type[X], count, offset):
        ArrayType = c_struct * int(count)
        entries_ctypes = self.read_struct(ArrayType, offset)
        entries = np.ctypeslib.as_array(entries_ctypes)
        return entries

    def _col_prisms(self, start_ptr, end_ptr) -> _PrismEntries:
        count = (end_ptr.value - (start_ptr.value + 0x10)) // ctypes.sizeof(struct_kcol_prism_data_t)
        return cast(_PrismEntries, self._np_entries(struct_kcol_prism_data_t, count, start_ptr.value + 0x10))

    def _col_verts(self, ptr, count) -> Annotated[NDArray[np.float32], Literal["N", 3]]:
        return self._np_entries(VecFx32, count, ptr.value)

    def _col_nrms(self, ptr, count) -> Annotated[NDArray[np.float32], Literal["N", 3]]:
        return self._np_entries(VecFx16, count, ptr.value)

    
    
    def _col_entries(self) -> _CollisionEntries:
        header = self.collision_header
        prism = self._col_prisms(header.prismDataOffset, header.blockDataOffset) # triangular prisms
        prism_attribute = _unpack_col_attributes(prism['attribute'])
        vert = self._col_verts(header.posDataOffset, prism['posIdx'].max() + 1) # vertices
        nrm_ids = np.stack([prism['fNrmIdx'], prism['eNrm1Idx'], prism['eNrm2Idx'], prism['eNrm3Idx']])
        nrm = self._col_nrms(header.nrmDataOffset, nrm_ids.max() + 1) # normals

        return {
            'prism': prism,
            'prism_attribute': prism_attribute,
            'vert': vert,
            'nrm': nrm,
        }
    
    
    def _col_prism_vertices(self, 
                            prisms: _PrismEntries, 
                            positions: Annotated[NDArray[np.float32], Literal["N", 3]], 
                            normals: Annotated[NDArray[np.float32], Literal["N", 3]]) -> _TriangleVertices:
        height = cast(NDArray[np.float32], prisms['height']['val']) / (1 << 12)
        v1 = _pack_i4_fx32(positions[prisms['posIdx']])
        fNrm = _pack_i4_fx32(normals[prisms['fNrmIdx']])
        eNrm1 = _pack_i4_fx32(normals[prisms['eNrm1Idx']])
        eNrm2 = _pack_i4_fx32(normals[prisms['eNrm2Idx']])
        eNrm3 = _pack_i4_fx32(normals[prisms['eNrm3Idx']])

        crossA = np.cross(eNrm1, fNrm, axis=-1)
        crossB = np.cross(eNrm2, fNrm, axis=-1)

        v2: np.ndarray = (
            v1
            + crossB * (height / np.vecdot(eNrm3, crossB))[:, None]
        )
        v3: np.ndarray = (
            v1
            + crossA * (height / np.vecdot(eNrm3, crossA))[:, None]
        )

        return {
            "v1": torch.tensor(v1, dtype=torch.float32),
            "v2": torch.tensor(v2, dtype=torch.float32),
            "v3": torch.tensor(v3, dtype=torch.float32)
        }

    @property
    def collision_data(self) -> CollisionData:
        """
        Retrieve collision data from the KCL (Collision) file.

        Lazily initializes and caches collision data on first access. Parses collision
        entries, prism vertices, and normal vectors from the KCL data structure.

        Returns
        -------
        `CollisionData`: A dictionary containing collision entries, prism data,
                          vertex positions, and normal vectors.
        """
        if self._kcl_data is None:
            entries = self._col_entries()
            triangles = self._col_prism_vertices(entries['prism'], entries['vert'], entries['nrm'])
            self._kcl_data = {
                **entries,
                **triangles
            }

        return cast(CollisionData, self._kcl_data)

    @property
    def driver_status(self):
        """
        Get the status of the primary driver (driver 0).
        
        Returns
        -------
        `dict`: A dictionary containing the status information of driver 0,
                  including position, speed, lap count, and other driver-related data.
        """
        return self.get_driver_status(0)

    @property
    def frame_count_race_start(self) -> int:
        """
        Get the frame count since the start of the race.
        
        Returns
        -------
        `int`: The current frame counter value from the race status.
        """
        return self.race_status.time.frameCounter

    @property
    def frame_count(self) -> int:
        """
        Get the current frame count of the race.
        
        Returns
        -------
        `int`: The frame counter value from the current race state.
        """
        return self.race_state.frameCounter

    def _octree_search(self, point: tuple[float, float, float]):
        header = self.collision_header

        px, py, pz = point
        minx, miny, minz = header.areaMinPos

        x = int(px - (minx / (1 << 12)))
        if (x & header.areaXWidthMask) != 0:
            return None

        y = int(py - (miny / (1 << 12)))
        if (y & header.areaYWidthMask) != 0:
            return None

        z = int(pz - (minz / (1 << 12)))
        if (z & header.areaZWidthMask) != 0:
            return None

        block_start: int = header.blockDataOffset.value
        node_count = (len(self.memoryview) - block_start) // 4
        NodeArrayType = OctreeNode * node_count
        nodes = self.read_struct(NodeArrayType, block_start)

        # initialize root
        shift = header.blockWidthShift
        cur_node_idx = 0  # root at start of block_data

        child_idx = (((z >> shift) << header.areaXYBlockShift) |
                    ((y >> shift) << header.areaXBlockShift) |
                    (x >> shift))

        while True:
            node = nodes[cur_node_idx + child_idx]

            if node.is_leaf:
                # negative flag = leaf node
                return block_start + (cur_node_idx * 4) + node.offset

            cur_node_idx += (node.offset // 4)
            shift -= 1

            # initialize next index
            child_idx = (((z >> shift) & 1) << 2 |
                        ((y >> shift) & 1) << 1 |
                        ((x >> shift) & 1))

    def collision_search(self, point: tuple[float, float, float]):
        block_data_offset = self.collision_header.blockDataOffset
        leaf_offset = self._octree_search(point)

        tri_indices: list[int] = []
        start = block_data_offset + leaf_offset + 2
        entry_count = (len(self.get_memoryview()) - start) // 2
        ChunkArrayType = ctypes.c_uint16 * entry_count
        chunks = self.read_struct(ChunkArrayType, start)
        for val in chunks:
            if val == 0:
                break

            tri_indices.append(val - 1)

        if len(tri_indices) == 0:
            return None

        return tri_indices

    def get_driver_status(self, index) -> struct_race_driver_status_t:
        """
        Retrieve the status of a driver at the specified index.
        
        Args
        ----
        index: The index of the driver in the race status array.
        
        Returns
        -------
        `struct_race_driver_status_t`: A structure containing the driver's current status information.
        """
        return self.race_status.driverStatus[index]

    # PyTorch API Methods #
    def set_torch_device(self, device: torch.device):
        self._device = device

    def get_torch_device(self):
        """
        Get the torch device used for tensor operations.
        
        Returns
        -------
        `torch`.device: The device (CPU, CUDA, or MPS) configured for this emulator instance.
        """
        return self._device

    # Memory API Methods #
    def get_camera_settings(self):
        """
        Retrieve the current camera settings.
        
        Returns
        -------
        `dict`: A dictionary containing the camera configuration parameters:
                - fov_sin (float): Sine of the field of view angle.
                - fov_cos (float): Cosine of the field of view angle.
                - aspect (float): The aspect ratio of the camera viewport.
                - far (float): The far clipping plane distance.
                - near (float): The near clipping plane distance.
        """
        return {
            "fov_sin": self.camera.fovSin,
            "fov_cos": self.camera.fovCos,
            "aspect": self.camera.aspectRatio,
            "far": self.camera.frustumFar,
            "near": self.camera.frustumNear,
        }

    # Screen API Methods #
    def _mv_matrix(self):
        device = self.get_torch_device()

        out = torch.eye(4).to(device)
        mtx = self.camera.mtx.to(device).T
        mtx[:3, 3] *= 16 # position is scaled by 16
        out[:3, :] = mtx

        return out

    def _proj_matrix(self):
        device = self.get_torch_device()

        cam = self.get_camera_settings()

        # opengl projection matrix
        out = torch.zeros((4, 4), device=device)
        out[0, 0] = cam['fov_cos'] / (cam['fov_sin'] * cam['aspect'])
        out[1, 1] = cam['fov_cos'] / cam['fov_sin']
        out[2, 2] = -(cam['far'] + cam['near']) / (cam['far'] - cam['near'])
        out[2, 3] = -(2 * cam['far'] * cam['near']) / (cam['far'] - cam['near'])
        out[3, 2] = -1

        return out

    def _convert_to_camera_space(self, points: torch.Tensor):
        mvm = self._mv_matrix()
        padded = torch.nn.functional.pad(points, (0, 1), "constant", 1)
        cam_space = (mvm @ padded.T).T # convert to camera space
        return cam_space

    def _convert_to_screen_space(self, points: torch.Tensor):
        pm = self._proj_matrix()
        far = self.get_camera_settings()['far']
        cam_space = self._convert_to_camera_space(points) # convert to camera space

        # convert to clip space
        clip_space = (pm @ cam_space.T).T

        # depth
        ndc = clip_space[:, :3] / clip_space[:, 3, None] # normalize w/ respect to w (new shape: (B, 3))

        # screen space
        screen_x = (ndc[:, 0] + 1) / 2 * SCREEN_WIDTH
        screen_y = (1 - ndc[:, 1]) / 2 * SCREEN_HEIGHT # Both DS screens are stitched into one. we only consider the top screen height

        screen_depth = clip_space[:, 2]

        return {
            'screen': torch.stack([screen_x, screen_y], dim=-1),
            'depth': screen_depth,
        }

    def _get_screen_z_clip_mask(self, screen_space):
        cam = self.get_camera_settings()
        z_clip: torch.Tensor = (screen_space[:, 2] > cam['near']) & (screen_space[:, 2] < cam['far'])

        return z_clip

    def project_to_screen(self, points: torch.Tensor, normalize_depth = False) -> dict[str, torch.Tensor]:
        """
        Project 3D points to 2D screen space with optional depth normalization.
        
        Args
        ----
        points (torch.Tensor): 3D points to project to screen space.
            normalize_depth (bool, optional): Whether to normalize depth values using the camera's far plane. Defaults to False.
        
        Returns
        -------
        `dict[str, torch.Tensor]`: Dictionary containing:
                - 'screen': Tensor with projected 2D screen coordinates and normalized/raw depth values.
                - 'mask': Boolean mask indicating which points are within the screen's z-clipping bounds.
        """
        sd = self._convert_to_screen_space(points)
        out = torch.cat([sd['screen'], sd['depth'][:, None]], dim=-1)
        mask = self._get_screen_z_clip_mask(out)
        if normalize_depth:
            far = self.get_camera_settings()['far']
            out[:, 2] = -far / (-far + out[:, 2])

        return {
            'screen': out,
            'mask': mask
        }

    
    def checkpoint_pos(self, device=None) -> _CheckpointPos:
        curr_checkpoint_id = self.driver_status.curCpoi
        next_checkpoint_id = curr_checkpoint_id + 1 if curr_checkpoint_id + 1 < self.map_data.cpoiCount else 0
        curr_entry = self.checkpoint_data[curr_checkpoint_id]
        next_entry = self.checkpoint_data[next_checkpoint_id]
        _y = self.camera.target[1].item()

        curr_checkpoint_pos = torch.tensor([
            [curr_entry.x1, _y, curr_entry.z1],
            [curr_entry.x2, _y, curr_entry.z2]
        ], dtype=torch.float32, device=device)

        next_checkpoint_pos = torch.tensor([
            [next_entry.x1, _y, next_entry.z1],
            [next_entry.x2, _y, next_entry.z2]
        ], dtype=torch.float32, device=device)


        out: _CheckpointPos = {
            'current_checkpoint_id': curr_checkpoint_id,
            'current_checkpoint_pos': curr_checkpoint_pos,
            'next_checkpoint_id': next_checkpoint_id,
            'next_checkpoint_pos': next_checkpoint_pos,
        }

        return out

    
    def checkpoint_angle(self, device=None) -> _CheckpointAngle:
        pos_info = self.checkpoint_pos(device)
        C = pos_info['next_checkpoint_pos']
        mp = (C[0, :] - C[1, :]) / 2 # midpoint
        M = self.driver.mainMtx.to(device)[:3, :]
        mp_local = mp @ M.T
        mp_angle = torch.atan2(mp_local[2], mp_local[0])
        out: _CheckpointAngle = {
            'midpoint_angle': mp_angle
        }
        return out


    
    def checkpoint_info(self, device=None) -> CheckpointInfo:
        pos_info = self.checkpoint_pos(device)
        angle_info = self.checkpoint_angle(device)
        out: CheckpointInfo = {
            **pos_info,
            **angle_info
        }
        return out




    def read_facing_point_checkpoint(self, device=None):
        position = self.driver.position
        direction = self.driver.direction
        checkpoint = self.checkpoint_info()['next_checkpoint_pos']
        mask_xz = torch.tensor([0, 2], dtype=torch.int32, device=device)
        pos_xz = position[mask_xz]
        dir_xz = direction[mask_xz]
        pxz_1, pxz_2 = checkpoint[:, mask_xz].chunk(2, dim=0)
        pxz_1 = pxz_1.squeeze(0)
        pxz_2 = pxz_2.squeeze(0)
        intersect, _ = intersect_ray_line_2d(pos_xz, dir_xz, pxz_1, pxz_2)
        intersect = torch.tensor([intersect[0], position[1], intersect[1]], device=device)
        return intersect

    def obstacle_info(self, n_rays, max_dist=float('inf'), device=None) -> RayCastInfo:
        M = self.driver.mainMtx.to(device)[:3, :].T
        pos = self.driver.position.to(device)
        _, R = generate_plane_vectors(n_rays, 180, M, pos)
        pos[1] += 10.0
        pos = pos.unsqueeze(0)
        col_data = self.collision_data
        wall_mask = col_data['prism_attribute']['is_floor'] != 1

        v1 = col_data['v1'].to(device)
        v2 = col_data['v2'].to(device)
        v3 = col_data['v3'].to(device)
        V = torch.stack([v1, v2, v3], dim=1)
        V = V[wall_mask, :, :] # (B, 3, 3)
        B = R.shape[0]
        P = pos.repeat(B, 1)



        all_hits = vmap_all_pairs(V, P, R, False, 1e-6)
        distances = all_hits[:, :, 0]
        min_dists, hit_ids = torch.min(torch.nan_to_num(distances, nan=float('inf')), dim=1)
        valid_rays_mask = min_dists < max_dist
        min_dists[~valid_rays_mask] = max_dist

        out: RayCastInfo = {
            "distance": min_dists,
            "position": P + (R * min_dists[:, None]),
            "mask": valid_rays_mask,
        }

        return out

    def get_obs(self, n_rays, max_dist, device=None):
        return torch.cat([
            self.obstacle_info(n_rays, max_dist=max_dist, device=device)['distance'],
            self.checkpoint_info(device)['midpoint_angle'].reshape(1)
        ], dim=-1)


MT = TypeVar('MT', bound=Union[np.ndarray, list])
class Metadata(TypedDict, Generic[MT]):
    mean: MT
    std: MT
    size: int

def _combine(m1: np.ndarray, m2: np.ndarray, std1: np.ndarray, std2: np.ndarray, n1: int, n2: int):
    mean = (n1 * m1 + n2 * m2) / (n1 + n2)
    sn1 = (n1 - 1) * std1**2
    sn2 = (n2 - 1) * std2**2
    sn3 = n1 * n2 / (n1 + n2)
    sn4 = m1**2 + m2**2 - (2 * m1 * m2)
    sd = n1 + n2 - 1
    std = ((sn1 + sn2 + sn3 * sn4) / sd)**0.5
    return mean, std, n1 + n2

def _to_numpy(mdata: Metadata[list]) -> Metadata[np.ndarray]:
    return {
        "mean": np.array(mdata['mean']),
        "std": np.array(mdata['std']),
        "size": mdata['size']
    }

def _to_list(mdata: Metadata[np.ndarray]) -> Metadata[list]:
    return {
        "mean": mdata['mean'].tolist(),
        "std": mdata['std'].tolist(),
        "size": mdata['size']
    }

def combine(*mdata: Metadata):
    assert len(mdata) != 0


    if isinstance(mdata[0]['mean'], list):
        out = _to_numpy(mdata[0])
    else:
        out = mdata[0]

    if len(mdata) == 1:
        return _to_list(out)

    for m in mdata[1:]:
        _m = m if isinstance(m['mean'], np.ndarray) else _to_numpy(m)
        mean, std, size = _combine(
            out['mean'],
            _m['mean'],
            out['std'],
            _m['std'],
            out['size'],
            _m['size']
        )
        out['mean'] = mean
        out['std'] = std
        out['size'] = size

    return out




class FileIO:
    def __init__(self, sf: BytesIO, tf: BytesIO, mf: BytesIO):
        # File handles
        self.sf = sf # samples
        self.tf = tf # targets
        self.mf = mf # metadata

        # Metadata
        self.obs_dim: int | None = None
        self.mean: np.ndarray | None = None
        self._mean_sq: np.ndarray | None = None
        self.size: int = 0

    def write(self, x: np.ndarray, y: np.ndarray):
        if self.obs_dim is None:
            self.obs_dim = x.shape[-1]
        else:
            assert self.obs_dim == x.shape[-1]

        if self.mean is None:
            assert self.obs_dim is not None
            self.mean = np.zeros((self.obs_dim,))

        if self._mean_sq is None:
            assert self.obs_dim is not None
            self._mean_sq = np.zeros((self.obs_dim,))

        self.size += 1
        delta = x - self.mean
        self.mean += delta / self.size
        delta2 = x - self.mean
        self._mean_sq += delta * delta2

        self.sf.write(x.astype("float32").tobytes())
        self.tf.write(y.astype("int32").tobytes())

    @property
    def metadata(self) -> Metadata[np.ndarray]:
        std_dev = None
        if self.size > 1:
            assert self._mean_sq is not None
            variance = self._mean_sq / self.size
            std_dev = np.sqrt(variance)
        elif self.mean is not None:
            # Fallback if recording was empty
            std_dev = np.ones_like(self.mean)

        assert std_dev is not None and self.mean is not None

        return {
            "mean": self.mean,
            "std": std_dev,
            "size": self.size
        }

    def close(self, old_metadata: Optional[Metadata]):
        new_metadata = combine(self.metadata, old_metadata) if old_metadata is not None else self.metadata
        if isinstance(new_metadata['mean'], np.ndarray):
            new_metadata = _to_list(cast(Metadata[np.ndarray], new_metadata))

        new_metadata = cast(Metadata[list], new_metadata)
        json.dump(new_metadata, self.mf)
        self.sf.close()
        self.tf.close()
        self.mf.close()




class MarioKart(DeSmuME):
    """
    MarioKart emulator class that extends DeSmuME with ML-specific functionality.

    This class wraps a Mario Kart emulator to support machine learning training by:
    - Capturing game observations (rays) as input vectors
    - Recording input/output pairs for supervised learning
    - Computing gradients of observations across frames
    - Managing file I/O for dataset collection

    Attributes
    ----------
    `device` (`torch.device`): PyTorch device for tensor computations (cpu or cuda).\n
    `has_grad` (`bool`): Whether to compute and include observation gradients.\n
    `n_rays` (`int`): Number of ray-cast sensors for observation generation.\n
    `max_dist` (`float`): Maximum distance for ray-cast sensors.\n
    `count` (`int`): Frame counter since emulator initialization.\n
    `memory` (`MarioKart_Memory`): Custom memory interface for game state access.
        
    Examples
    --------
    ```python
    mk = MarioKart(rom_path, n_rays=8, has_grad=True, device=torch.device("cuda"))
    mk.enable_file_io(state_file, trace_file, metadata_file)
    for _ in range(1000):
        mk.cycle(with_joystick=False)
    mk.close()
    ```
    """
    def __init__(self, *args, n_rays=NUM_RAYS, has_grad=False, max_dist=MAX_DIST, device: Optional[torch.device]=None, **kwargs):
        """
        Initialize the emulator with configuration parameters.

        Args
        ----
        `*args`: Variable length positional arguments passed to parent class and MarioKart_Memory.\n
        `n_rays` (`int`): Number of rays for raycasting. Defaults to NUM_RAYS.\n
        `has_grad` (`bool`): Whether to enable gradient computation. Defaults to False.\n
        `max_dist` (`float`): Maximum distance for raycasting. Defaults to MAX_DIST.\n
        `device` (`Optional[torch.device]`): The device to run computations on (cpu, cuda, or mps).
        Defaults to cpu if None.\n
        `**kwargs`: Variable length keyword arguments passed to parent class and MarioKart_Memory.

        Attributes
        ----------
        `device` (`torch.device`): The compute device for tensor operations.\n
        `has_grad` (`bool`): Flag indicating gradient computation status.\n
        `n_rays` (`int`): Number of raycasting rays.\n
        `max_dist` (`float`): Maximum raycasting distance.\n
        `count` (`int`): Frame counter, initialized to 0.
        """
        super().__init__(*args, **kwargs)
        if device is None:
            device = torch.device("cpu")

        self.device = device
        self.has_grad = has_grad
        self._grad = None
        self.x0 = None
        self.y0 = None
        self.n_rays = n_rays
        self.max_dist = max_dist
        self._io = None
        self._max_frame = 0
        self.count = 0
        self._old_metadata: Optional[Metadata[list]] = None
        self._memory = MarioKart_Memory(self, *args, device=device, **kwargs)

    @property
    @override
    def memory(self) -> MarioKart_Memory:
        return self._memory

    @override
    def cycle(self, with_joystick=False):
        if self.has_grad and self.memory.race_ready:
            self._cycle_with_grad(with_joystick)
        else:
            super().cycle(with_joystick)

        self.count += 1

        if isinstance(self._io, FileIO) and self.memory.race_ready:
            if self.memory.frame_count_race_start <= self._max_frame:
                return

            input_vector = self.memory.get_obs(self.n_rays, max_dist=self.max_dist, device=self.device)
            if self.has_grad:
                input_vector = torch.cat([
                    input_vector,
                    self.grad
                ], dim=-1)

            x = input_vector.detach().cpu().numpy()

            keymask = self.input.keypad_get()
            y = np.array([keymask])

            self._io.write(x, y)
            self._max_frame = self.memory.frame_count_race_start



    @property
    def grad(self):
        if self._grad is None:
            device = self.memory._device
            y0 = self.memory.get_obs(self.n_rays, self.max_dist, device)
            self._grad = torch.zeros_like(y0)

        return self._grad


    def _cycle_with_grad(self, with_joystick):
        if self.x0 is None or self.y0 is None:
            self.x0 = self.memory.frame_count_race_start
            self.y0 = self.memory.get_obs(self.n_rays, self.max_dist, self.device)

        super().cycle(with_joystick)
        x1 = self.memory.frame_count_race_start
        y1 = self.memory.get_obs(self.n_rays, self.max_dist, self.device)
        self._grad = (y1 - self.y0) / (x1 - self.x0)
        self.x0, self.y0 = x1, y1

    def enable_file_io(self, sf, tf, mf):
        self._io = FileIO(sf, tf, mf)

    @override
    def reset(self):
        super().reset()
        self.memory._ready = False


    @override
    def close(self):
        if isinstance(self._io, FileIO):
            self._io.close(self._old_metadata)

        super().close()
