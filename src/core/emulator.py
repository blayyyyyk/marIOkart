from ctypes import Structure
import ctypes
from desmume.emulator import (
    DeSmuME,
    DeSmuME_Memory as DM,
    SCREEN_WIDTH,
    SCREEN_HEIGHT
)
from typing import Type, TypeVar, cast, Union, TypedDict, Literal, Generic, cast, Optional
from typing_extensions import override
from private.mkdslib import *
import numpy as np
import torch
from src.utils.vector import *
from numpy.lib.recfunctions import structured_to_unstructured
import math
import json
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

def _unpack_col_attributes(raw_attrs: np.ndarray):
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
    
    return result

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

def _col_prism_vertices(prisms: np.ndarray, positions: np.ndarray, normals: np.ndarray):
    height: np.ndarray = prisms['height']['val'] / (1 << 12)
    v1: np.ndarray = _pack_i4_fx32(positions[prisms['posIdx']])
    fNrm: np.ndarray = _pack_i4_fx32(normals[prisms['fNrmIdx']])
    eNrm1: np.ndarray = _pack_i4_fx32(normals[prisms['eNrm1Idx']])
    eNrm2: np.ndarray = _pack_i4_fx32(normals[prisms['eNrm2Idx']])
    eNrm3: np.ndarray = _pack_i4_fx32(normals[prisms['eNrm3Idx']])

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


class DeSmuME_Memory(DM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.emu.lib is not None
        self.memoryview = memoryview(MMUPrefix.in_dll(self.emu.lib, "MMU").MAIN_MEM)
        
    def read_struct(self, struct_t: Type[T], addr: int) -> T:
        return DeSmuME_Memory._read_struct(self.memoryview, struct_t, addr) # main memory region 
    
    def get_memoryview(self):
        return self.memoryview
    
    @staticmethod
    def _read_struct(mem, struct_t: Type[T], addr: int) -> T:
        struct = struct_t.from_buffer(mem, addr - 0x02000000) # main memory region 
        return struct


class MarioKart_Memory(DeSmuME_Memory):
    def __init__(self, *args, device: torch.device, **kwargs):
        super().__init__(*args, **kwargs)
        self._driver = None
        self._camera = None
        self._race_status = None
        self._map_data = None
        self._cpoi_data = None
        self._kcl_header = None
        self._kcl_data = None
        self._race_state = None
        self._ready = False
        self._device = device

    @property
    def driver(self) -> driver_t:
        if self._driver is None:
            driver_addr = self.unsigned.read_long(RACER_PTR_ADDR)
            self._driver = self.read_struct(driver_t, driver_addr)

        return self._driver
    
    @property
    def camera(self) -> camera_t:
        if self._camera is None:
            camera_addr = self.unsigned.read_long(CAMERA_PTR_ADDR)
            self._camera = self.read_struct(camera_t, camera_addr)

        return self._camera
    
    @property
    def race_status(self) -> race_status_t:
        if self._race_status is None:
            race_status_addr = self.unsigned.read_long(RACE_STATUS_PTR_ADDR)
            self._race_status = self.read_struct(race_status_t, race_status_addr)

        return self._race_status
    
    @property
    def map_data(self) -> mdat_mapdata_t:
        if self._map_data is None:
            map_data_addr = self.unsigned.read_long(MAP_DATA_PTR_ADDR)
            self._map_data = self.read_struct(mdat_mapdata_t, map_data_addr)
            
        return self._map_data
    
    @property
    def checkpoint_data(self) -> ctypes.Array[struct_nkm_cpoi_entry_t]:
        if self._cpoi_data is None:
            mdata = self.map_data
            CpoiArrayType = struct_nkm_cpoi_entry_t * int(mdata.cpoiCount)
            self._cpoi_data = self.read_struct(CpoiArrayType, mdata.cpoi.value)

        return self._cpoi_data
    
    @property
    def collision_header(self) -> kcol_header_t:
        if self._kcl_header is None:
            self._kcl_header = self.read_struct(kcol_header_t, COLLISION_DATA_ADDR)
        
        return self._kcl_header

    @property
    def race_state(self) -> race_state_t:
        if self._race_state is None:
            addr = self.unsigned.read_long(RACE_STATE_PTR_ADDR)
            self._race_state = self.read_struct(race_state_t, addr)

        return self._race_state
    
    @property
    def race_ready(self):
        if not self._ready:
            try:
                _f = self.race_state.frameCounter - self.race_state.frameCounter2
                self._ready = _f == 1
            except:
                pass

        return self._ready

    def _np_entries(self, c_struct, count, offset):
        ArrayType = c_struct * int(count)
        entries_ctypes = self.read_struct(ArrayType, offset)
        entries = np.ctypeslib.as_array(entries_ctypes)
        return entries
    
    def _col_prisms(self, start_ptr, end_ptr):
        count = (end_ptr.value - (start_ptr.value + 0x10)) // ctypes.sizeof(struct_kcol_prism_data_t)
        return self._np_entries(struct_kcol_prism_data_t, count, start_ptr.value + 0x10)
    
    def _col_verts(self, ptr, count):
        return self._np_entries(VecFx32, count, ptr.value)
    
    def _col_nrms(self, ptr, count):
        return self._np_entries(VecFx16, count, ptr.value)
    
    def _col_entries(self):
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

    @property
    def collision_data(self):
        if self._kcl_data is None:
            entries = self._col_entries()
            triangles = _col_prism_vertices(entries['prism'], entries['vert'], entries['nrm'])
            self._kcl_data = {
                **entries,
                **triangles
            }

        return self._kcl_data
    
    @property
    def driver_status(self):
        return self.get_driver_status(0)
    
    @property
    def frame_count_race_start(self) -> int:
        return self.race_status.time.frameCounter
    
    @property
    def frame_count(self) -> int:
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

        block_start = header.blockDataOffset
        node_count = (len(self.get_memoryview()) - block_start) // 4
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
        return self.race_status.driverStatus[index]
    
    # PyTorch API Methods #
    def set_torch_device(self, device: torch.device):
        self._device = device

    def get_torch_device(self):
        return self._device
    
    # Memory API Methods #
    def get_camera_settings(self):
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

    CheckpointPos = TypedDict('CheckpointPos', {
        'current_checkpoint_id': int,
        'current_checkpoint_pos': torch.Tensor,
        'next_checkpoint_id': int,
        'next_checkpoint_pos': torch.Tensor,
    })
    def checkpoint_pos(self, device=None) -> CheckpointPos:
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


        out: MarioKart_Memory.CheckpointPos = {
            'current_checkpoint_id': curr_checkpoint_id,
            'current_checkpoint_pos': curr_checkpoint_pos,
            'next_checkpoint_id': next_checkpoint_id,
            'next_checkpoint_pos': next_checkpoint_pos,
        }
        
        return out
    
    CheckpointAngle = TypedDict('CheckpointAngle', {
        'midpoint_angle': torch.Tensor
    })
    def checkpoint_angle(self, device=None) -> CheckpointAngle:
        pos_info = self.checkpoint_pos(device)
        C = pos_info['next_checkpoint_pos']
        mp = (C[0, :] - C[1, :]) / 2 # midpoint
        M = self.driver.mainMtx.to(device)[:3, :]
        mp_local = mp @ M.T
        mp_angle = torch.atan2(mp_local[2], mp_local[0])
        out: MarioKart_Memory.CheckpointAngle = {
            'midpoint_angle': mp_angle
        }
        return out


    class CheckpointInfo(CheckpointPos, CheckpointAngle): pass
    def checkpoint_info(self, device=None) -> CheckpointInfo:
        pos_info = self.checkpoint_pos(device)
        angle_info = self.checkpoint_angle(device)
        out: MarioKart_Memory.CheckpointInfo = {
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
    
    RayCastInfo = TypedDict('RayCastInfo', {'distance': torch.Tensor, 'position': torch.Tensor, 'mask': torch.Tensor})
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

        out: MarioKart_Memory.RayCastInfo = {
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
    std = (sn1 + sn2 + sn3 * sn4 / sd)**0.5
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
    def __init__(self, *args, n_rays=NUM_RAYS, has_grad=False, max_dist=MAX_DIST, device=None, **kwargs):
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
        self._old_metadata: Metadata[list] | None = None
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