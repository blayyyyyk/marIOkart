import torch
from utils.kcl_torch import KCLTensor as KCL
from utils.nkm_torch import NKMTensor as NKM
from mkds.utils import read_fx16, read_u16, read_u32, read_fx32, read_vector_3d_fx32
from mkds.utils import read_s32
from desmume.emulator import MemoryAccessor
from utils.vector import compute_model_view, project_to_screen, triangle_raycast_batch, sample_cone
import math

RACER_DATA_PTR = 0x0217ACF8
CAMERA_DATA_PTR = 0x0217AA4C
OBJECT_DATA_PTR = 0x0217B588

# Object flags
FLAG_DYNAMIC = 0x1000
FLAG_MAPOBJ  = 0x2000
FLAG_ITEM    = 0x4000
FLAG_RACER   = 0x8000

class Racer:
    def __init__(self, kcl: KCL, nkm: NKM, z_near, z_far, device=None):
        self.device = device
        self.kcl = kcl
        self.nkm = nkm
        self.racer_data_addr: int | None = None
        self.camera_data_addr: int | None = None
        self.camera_z_near = z_near
        self.camera_z_far = z_far
        self.memory: MemoryAccessor | None = None
        
        
    @classmethod
    def from_path(cls, kcl_path: str, nkm_path: str, z_near=0.0, z_far=100.0, device=None):
        kcl = KCL.from_file(kcl_path, device=device)   
        nkm = NKM.from_file(nkm_path, device=device)
        return cls(kcl, nkm, z_near, z_far, device=device)
        
    @property
    def position(self) -> torch.Tensor:
        if self.racer_data_addr is None:
            assert self.memory is not None
            self.racer_data_addr = read_u32(self.memory, RACER_DATA_PTR)
        
        pos = read_vector_3d_fx32(self.memory, self.racer_data_addr + 0x80)
        return torch.tensor(pos, device=self.device)
        
    @property
    def direction(self) -> torch.Tensor:
        if self.racer_data_addr is None:
            assert self.memory is not None
            self.racer_data_addr = read_u32(self.memory, RACER_DATA_PTR)
            
        pos = read_vector_3d_fx32(self.memory, self.racer_data_addr + 0x68)
        return torch.tensor(pos, device=self.device)
        
    @property
    def camera_details(self) -> tuple[float, float]:
        if self.camera_data_addr is None:
            assert self.memory is not None
            self.camera_data_addr = read_u32(self.memory, CAMERA_DATA_PTR)
            
        fov = read_u16(self.memory, self.camera_data_addr + 0x60) * (2 * math.pi / 0x10000)
        aspect = read_fx32(self.memory, self.camera_data_addr + 0x6C)
        return fov, aspect
        
        
    # TODO
    @property
    def objects(self):
        obj_array_ptr = read_s32(self.memory, OBJECT_DATA_PTR + 0x10)
        array_offset = obj_array_ptr + 1
        max_count = read_u16(self.memory, OBJECT_DATA_PTR + 0x08)
        count = 0
        idx = 0
        while count < max_count:
            current = array_offset + idx * 0x1C
            obj_ptr = read_u32(self.memory, current + 0x18)
            flags = read_u16(self.memory, current + 0x14)
            if obj_ptr == 0:
                continue
                
            count += 1
            
            if flags & 0x200 != 0:
                continue
                
            pos_ptr = read_s32(self.memory, current + 0x0C)
            if pos_ptr == 0:
                continue
                
                
            if flags & FLAG_RACER != 0:
                pass
            
        
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
            self.camera_data_addr = read_u32(self.memory, CAMERA_DATA_PTR)
            
        pos = read_vector_3d_fx32(self.memory, self.camera_data_addr + 0x24)
        elevation = read_fx32(self.memory, self.camera_data_addr + 0x178)
        pos = (pos[0], pos[1], pos[2])
        return torch.tensor(pos, device=self.device)
        
    @property
    def camera_target_position(self) -> torch.Tensor:
        if self.camera_data_addr is None:
            assert self.memory is not None
            self.camera_data_addr = read_u32(self.memory, CAMERA_DATA_PTR)
            
        data_addr = read_u32(self.memory, CAMERA_DATA_PTR)
        pos = read_vector_3d_fx32(self.memory, data_addr + 0x18)
        return torch.tensor(pos, device=self.device)
        
    @property
    def model_view(self) -> torch.Tensor:
        return compute_model_view(
            self.camera_position,
            self.camera_target_position,
            device=self.device
        )
        
    def project_to_screen(self, x: torch.Tensor) -> torch.Tensor:
        return project_to_screen(
            x,
            self.model_view, 
            self.camera_fov,
            self.camera_aspect,
            self.camera_z_far,
            self.camera_z_near,
            device=self.device
        )
    
    def get_facing_point(self, position: torch.Tensor, direction: torch.Tensor):
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
        
    def get_forward_distance(self):
        ray_point = self.get_facing_point(self.position, self.direction)
        if ray_point is None:
            return None
            
        dist = torch.sqrt(torch.sum((self.position - ray_point)**2, dim=0))
        return dist
        
    def get_left_distance(self):
        up_basis = -torch.tensor([0, 1.0, 0], device=self.device, dtype=torch.float32)
        left_basis = torch.cross(self.direction, up_basis)
        ray_point = self.get_facing_point(self.position, left_basis)
        if ray_point is None:
            return None
            
        dist = torch.sqrt(torch.sum((self.position - ray_point)**2, dim=0))
        return dist
        
    def get_right_distance(self):
        up_basis = torch.tensor([0, 1.0, 0], device=self.device, dtype=torch.float32)
        right_basis = torch.cross(self.direction, up_basis)
        ray_point = self.get_facing_point(self.position, right_basis)
        if ray_point is None:
            return None
            
        dist = torch.sqrt(torch.sum((self.position - ray_point)**2, dim=0))
        return dist
        
        
    def get_forward_facing_distance(self, pos):
        return self.kcl.search_triangles(pos)