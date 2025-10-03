import torch
from utils.kcl_torch import KCLTensor as KCL
from utils.nkm_torch import NKMTensor as NKM
from mkds.utils import read_fx16, read_u16, read_u32, read_fx32, read_vector_3d_fx32
from mkds.utils import read_s32
from desmume.emulator import MemoryAccessor
from utils.vector import compute_model_view
import math

RACER_DATA_PTR = 0x0217ACF8
CAMERA_DATA_PTR = 0x0217AA4C

class Racer:
    def __init__(self, kcl: KCL, nkm: NKM, device=None):
        self.device = device
        self.kcl = kcl
        self.nkm = nkm
        self.racer_data_addr: int | None = None
        self.camera_data_addr: int | None = None
        self.memory: MemoryAccessor | None = None
        
    @classmethod
    def from_path(cls, kcl_path: str, nkm_path: str, device=None):
        kcl = KCL.from_file(kcl_path, device=device)   
        nkm = NKM.from_file(nkm_path, device=device)
        return cls(kcl, nkm, device=device)
        
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
    def camera_fov(self) -> tuple[float, float]:
        if self.camera_data_addr is None:
            assert self.memory is not None
            self.camera_data_addr = read_u32(self.memory, CAMERA_DATA_PTR)
            
        fov = read_u16(self.memory, self.camera_data_addr + 0x60) * (2 * math.pi / 0x10000)
        aspect = read_fx32(self.memory, self.camera_data_addr + 0x6C)
        return fov, aspect
        
    @property
    def camera_position(self) -> torch.Tensor:
        if self.camera_data_addr is None:
            assert self.memory is not None
            self.camera_data_addr = read_u32(self.memory, CAMERA_DATA_PTR)
            
        pos = read_vector_3d_fx32(self.memory, self.camera_data_addr + 0x24)
        elevation = read_fx32(self.memory, self.camera_data_addr + 0x178)
        pos = (pos[0], pos[1], pos[2] + elevation)
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
    
    def get_forward_facing_distance(self, pos):
        return self.kcl.search_triangles(pos)