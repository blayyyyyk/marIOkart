import torch
from utils.kcl_torch import KCLTensor as KCL
from utils.nkm_torch import NKMTensor as NKM
from mkds.utils import read_fx16, read_s16, read_u16, read_u32, read_fx32, read_vector_3d_fx32, read_s8, read_u8
from mkds.utils import read_s32
from desmume.emulator import MemoryAccessor
from utils.vector import compute_model_view, extrapolate, project_to_screen, triangle_raycast_batch, sample_cone, intersect_ray_line_2d
import math

from utils.object import DynamicObject, GameObject, MapObject, RacerObject, GameItemObject


OBJECT_DATA_PTR = 0x0217B588
CHECKPOINT_DATA_PTR = 0x021755FC

# Object flags
FLAG_DYNAMIC = 0x1000
FLAG_MAPOBJ  = 0x2000
FLAG_ITEM    = 0x4000
FLAG_RACER   = 0x8000

DEFAULT_RACER_ADDRS = {
    "racer_data_ptr": 0x0217ACF8,
    "camera_data_ptr": 0x0217AA4C,
    "object_data_ptr": 0x0217B588,
    "checkpoint_data_ptr": 0x021755FC
}

class Racer:
    def __init__(self, kcl: KCL, nkm: NKM, z_near=0.0, z_far=1000.0, z_scale=10.0, device=None):
        self.device = device
        self.kcl = kcl
        self.nkm = nkm
        self.racer_data_addr: int | None = None
        self.camera_data_addr: int | None = None
        self.object_data_addr: int | None = None
        self.checkpoint_data_addr: int | None = None
        self.camera_z_near = z_near
        self.camera_z_far = z_far
        self.camera_z_scale = z_scale
        self.memory: MemoryAccessor | None = None
        self.objects = {}
        self.z_clip_mask = lambda x: (x[:, 2] < -z_near) & (x[:, 2] > -z_far)
        self.checkpoint_cache = self.init_checkpoint_cache()
        
    def init_checkpoint_cache(self):
        floor_mask = self.kcl.prisms.is_floor == 1
        floor_points = self.kcl.triangles[floor_mask]
        floor_points = floor_points.reshape(floor_points.shape[0] * 3, 3)
        
        return torch.stack([
            extrapolate(self.nkm._CPOI.position1, floor_points, dim=1),
            extrapolate(self.nkm._CPOI.position2, floor_points, dim=1)
        ], dim=1)
        
    
        
        
    @property
    def course_id(self) -> int | None:
        if self.memory is None:
            return None
        return self.memory.read_byte(0x23cdcd8)
        
        
    @classmethod
    def from_path(cls, kcl_path: str, nkm_path: str, z_near=0.0, z_far=100.0, device=None):
        kcl = KCL.from_file(kcl_path, device=device)   
        nkm = NKM.from_file(nkm_path, device=device)
        return cls(kcl, nkm, z_near, z_far, device=device)
        
    @property
    def position(self) -> torch.Tensor:
        if self.racer_data_addr is None:
            assert self.memory is not None
            self.racer_data_addr = self.memory.read_long(DEFAULT_RACER_ADDRS['racer_data_ptr'])
        
        pos = read_vector_3d_fx32(self.memory, self.racer_data_addr + 0x80)
        return torch.tensor(pos, device=self.device)
        
    @property
    def direction(self) -> torch.Tensor:
        if self.racer_data_addr is None:
            assert self.memory is not None
            self.racer_data_addr = self.memory.read_long(DEFAULT_RACER_ADDRS['racer_data_ptr'])
            
        pos = read_vector_3d_fx32(self.memory, self.racer_data_addr + 0x68)
        return torch.tensor(pos, device=self.device)
        
    @property
    def camera_details(self) -> tuple[float, float]:
        if self.camera_data_addr is None:
            assert self.memory is not None
            self.camera_data_addr = self.memory.read_long(DEFAULT_RACER_ADDRS['camera_data_ptr'])
            
        fov = read_u16(self.memory, self.camera_data_addr + 0x60) * (2 * math.pi / 0x10000)
        aspect = read_fx32(self.memory, self.camera_data_addr + 0x6C)
        return fov, aspect
        
        
    def get_objects(self):
        obj_array_ptr = read_s32(self.memory, OBJECT_DATA_PTR + 0x10)
        array_offset = obj_array_ptr
        max_count = read_u16(self.memory, OBJECT_DATA_PTR + 0x08)
        count = 0
        idx = 0
        objects = {}
        while idx < 255 and count != max_count:
            obj = GameObject.from_bytes(idx, self.memory)
            if obj.flags & FLAG_MAPOBJ != 0:
                obj = MapObject(**vars(obj))
            elif obj.flags & FLAG_RACER != 0:
                obj = RacerObject(**vars(obj))
            elif obj.flags & FLAG_ITEM != 0:
                obj = GameItemObject(**vars(obj))
            elif obj.flags & FLAG_DYNAMIC == 0:
                obj = DynamicObject(**vars(obj))
                
            
                
            if obj.is_deleted:
                
                continue
                
            count += 1
            
            if obj.is_deactivated:
                
                continue
                
            if obj.is_removed:
                
                continue
                
            if obj.ptr not in objects:
                objects[obj.ptr] = obj
            
            
            idx += 1
            
        self.objects = objects
        return self.objects
        
    def _get_objects_details(self, cls):
        objects = self.get_objects()
        out = {}
        for key, obj in objects.items():
            if not isinstance(obj, cls):
                continue
                
            out[key] = obj
        return out
        
    def get_objects_details(self):
        return self._get_objects_details(GameObject)
    
    def get_map_objects_details(self):
        return self._get_objects_details(MapObject)
        
    def get_racer_objects_details(self):
        return self._get_objects_details(RacerObject)
        
    def get_item_objects_details(self):
        return self._get_objects_details(GameItemObject)
        
    def get_dynamic_objects_details(self):
        return self._get_objects_details(DynamicObject)
        
        
    @property
    def camera_fov(self) -> float:
        return self.camera_details[0]
        
    @property
    def camera_aspect(self) -> float:
        return self.camera_details[1]
        
    @property
    def camera_position(self) -> torch.Tensor:
        if self.camera_data_addr is None:
            assert self.memory is not None
            self.camera_data_addr = self.memory.read_long(DEFAULT_RACER_ADDRS['camera_data_ptr'])
            
        pos = read_vector_3d_fx32(self.memory, self.camera_data_addr + 0x24)
        elevation = read_fx32(self.memory, self.camera_data_addr + 0x178)
        pos = (pos[0], pos[1] + elevation, pos[2])
        return torch.tensor(pos, device=self.device)
        
    @property
    def camera_target_position(self) -> torch.Tensor:
        if self.camera_data_addr is None:
            assert self.memory is not None
            self.camera_data_addr = self.memory.read_long(DEFAULT_RACER_ADDRS['camera_data_ptr'])
            
        pos = read_vector_3d_fx32(self.memory, self.camera_data_addr + 0x18)
        return torch.tensor(pos, device=self.device)
        
    @property
    def model_view(self) -> torch.Tensor:
        return compute_model_view(
            self.camera_position,
            self.camera_target_position,
            device=self.device
        )
        
    
    def get_checkpoint_details(self):
        assert self.memory is not None
        if self.checkpoint_data_addr is None:
            self.checkpoint_data_addr = read_u32(self.memory, DEFAULT_RACER_ADDRS['checkpoint_data_ptr'])
            
        checkpoint = read_u8(self.memory, self.checkpoint_data_addr + 0x46)
        key_checkpoint = read_s8(self.memory, self.checkpoint_data_addr + 0x48)
        checkpoint_ghost = read_s8(self.memory, self.checkpoint_data_addr + 0xD2)
        key_checkpoint_ghost = read_s8(self.memory, self.checkpoint_data_addr + 0xD4)
        lap = read_s8(self.memory, self.checkpoint_data_addr + 0x38)
        
        checkpoint_count = self.nkm._CPOI.entry_count
        next_checkpoint = checkpoint + 1 if checkpoint + 1 != checkpoint_count else 0
     
        return {
            "checkpoint": checkpoint,
            "next_checkpoint": next_checkpoint,
            "keyCheckpoint": key_checkpoint,
            "checkpointGhost": checkpoint_ghost,
            "keyCheckpointGhost": key_checkpoint_ghost,
            "lap": lap
        }
           
    def get_current_checkpoint(self) -> torch.Tensor:
        checkpoint_details = self.get_checkpoint_details()
        checkpoint_idx = checkpoint_details['checkpoint']
        return self.checkpoint_cache[checkpoint_idx]
        
    def get_next_checkpoint(self) -> torch.Tensor:
        checkpoint_details = self.get_checkpoint_details()
        checkpoint_idx = checkpoint_details['next_checkpoint']
        return self.checkpoint_cache[checkpoint_idx]
        
    def project_to_screen(self, x: torch.Tensor) -> torch.Tensor:
        return project_to_screen(
            x,
            self.model_view, 
            self.camera_fov,
            self.camera_aspect,
            self.camera_z_far,
            self.camera_z_near,
            self.camera_z_scale,
            device=self.device
        )
        
    def get_facing_point_checkpoint(self, position: torch.Tensor, direction: torch.Tensor):
        checkpoint = self.get_next_checkpoint()
        pos = position
        mask_xz = torch.tensor([0, 2], dtype=torch.int32, device=self.device)
        pos_xz = pos[mask_xz]
        dir_xz = direction[mask_xz]
        pxz_1, pxz_2 = checkpoint[:, mask_xz].chunk(2, dim=0)
        pxz_1 = pxz_1.squeeze(0)
        pxz_2 = pxz_2.squeeze(0)
        intersect, _ = intersect_ray_line_2d(pos_xz, dir_xz, pxz_1, pxz_2)
        intersect = torch.tensor([intersect[0], position[1], intersect[1]], device=self.device)
        return intersect
        
    def get_forward_distance_checkpoint(self):
        ray_point = self.get_facing_point_checkpoint(self.position, self.direction)
        return torch.norm(ray_point - self.position)
        
    def get_left_distance_checkpoint(self):
        up_basis = -torch.tensor([0, 1.0, 0], device=self.device, dtype=torch.float32)
        left_basis = torch.cross(self.direction, up_basis)
        ray_point = self.get_facing_point_checkpoint(self.position, left_basis)
        return torch.norm(ray_point - self.position)
        
    def get_direction_checkpoint(self):
        f = self.get_forward_distance_checkpoint()
        l = self.get_left_distance_checkpoint()
        angle = torch.atan(f/l)
        return angle
        
    
    def get_facing_point_obstacle(self, position: torch.Tensor, direction: torch.Tensor):
        triangles = self.kcl.triangles
        wall_mask = self.kcl.prisms.is_wall == 1
        offroad_mask = (self.kcl.prisms.collision_type == 5) | (self.kcl.prisms.collision_type == 3) | (self.kcl.prisms.collision_type == 2)
        triangles = triangles[wall_mask | offroad_mask]
        B = triangles.shape[0]
        
        if B == 0:
            return
        
        v1, v2, v3 = triangles.chunk(3, dim=1)
        v1 = v1.squeeze(1)
        v2 = v2.squeeze(1)
        v3 = v3.squeeze(1)
        
        ray_dir = direction / torch.norm(direction, keepdim=True)
        angle = torch.tensor(torch.pi / 24, device=self.device)
        ray_samples = sample_cone(ray_dir, angle, 50)
        ray_dir = ray_dir.reshape(1, 3)
        ray_samples = torch.cat([ray_dir, ray_samples], dim=0)
        
        ray_origin = position
        ray_origin = ray_origin.unsqueeze(0)
        ray_origin = ray_origin.reshape(1, 3)
        ray_origin_samples = ray_origin.repeat(ray_samples.shape[0], 1)
        
        points = triangle_raycast_batch(ray_origin_samples, ray_samples, v1, v2, v3)
        N, M, C = points.shape
        points = points.reshape(N*M, C)
        if points.shape[0] == 0:
            return None
            
        dist = torch.cdist(points, ray_origin)
        min_id = torch.argmin(dist)
        current_point_min = points[min_id]
        return current_point_min
        
    
        
    def get_forward_distance_obstacle(self):
        ray_point = self.get_facing_point_obstacle(self.position, self.direction)
        if ray_point is None:
            return None
            
        dist = torch.sqrt(torch.sum((self.position - ray_point)**2, dim=0))
        return dist
        
    def get_left_distance_obstacle(self):
        up_basis = -torch.tensor([0, 1.0, 0], device=self.device, dtype=torch.float32)
        left_basis = torch.cross(self.direction, up_basis)
        ray_point = self.get_facing_point_obstacle(self.position, left_basis)
        if ray_point is None:
            return None
            
        dist = torch.sqrt(torch.sum((self.position - ray_point)**2, dim=0))
        return dist
        
    def get_right_distance_obstacle(self):
        up_basis = torch.tensor([0, 1.0, 0], device=self.device, dtype=torch.float32)
        right_basis = torch.cross(self.direction, up_basis)
        ray_point = self.get_facing_point_obstacle(self.position, right_basis)
        if ray_point is None:
            return None
            
        dist = torch.sqrt(torch.sum((self.position - ray_point)**2, dim=0))
        return dist
        

