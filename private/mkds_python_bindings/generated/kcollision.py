from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.types import *


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_collision_kcollision_h_3_9(Structure):
    _fields_ = [
        ('height', c_int),
        ('posIdx', c_int),
        ('fNrmIdx', c_int),
        ('eNrm1Idx', c_int),
        ('eNrm2Idx', c_int),
        ('eNrm3Idx', c_int),
        ('attribute', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_collision_kcollision_h_16_9(Structure):
    _fields_ = [
        ('posDataOffset', POINTER32(c_int)),
        ('nrmDataOffset', POINTER32(c_int)),
        ('prismDataOffset', POINTER32(kcol_prism_data_t)),
        ('blockDataOffset', POINTER32(c_int)),
        ('prismThickness', c_int),
        ('areaMinPos', c_int),
        ('areaXWidthMask', c_int),
        ('areaYWidthMask', c_int),
        ('areaZWidthMask', c_int),
        ('blockWidthShift', c_int),
        ('areaXBlocksShift', c_int),
        ('areaXYBlocksShift', c_int),
        ('sphereRadius', c_int),
    ]

class kcol_header_t(Structure):
    _fields_ = [
        ('posDataOffset', POINTER32(VecFx32)),
        ('nrmDataOffset', POINTER32(VecFx16)),
        ('prismDataOffset', POINTER32(kcol_prism_data_t)),
        ('blockDataOffset', POINTER32(u32)),
        ('prismThickness', fx32),
        ('areaMinPos', VecFx32),
        ('areaXWidthMask', u32),
        ('areaYWidthMask', u32),
        ('areaZWidthMask', u32),
        ('blockWidthShift', u32),
        ('areaXBlocksShift', u32),
        ('areaXYBlocksShift', u32),
        ('sphereRadius', fx32),
    ]

class kcol_prism_data_t(Structure):
    _fields_ = [
        ('height', fx32),
        ('posIdx', u16),
        ('fNrmIdx', u16),
        ('eNrm1Idx', u16),
        ('eNrm2Idx', u16),
        ('eNrm3Idx', u16),
        ('attribute', u16),
    ]
