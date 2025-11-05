from ctypes import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.types import *


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

class kcol_header_t(Structure):
    _fields_ = [
        ('posDataOffset', u32), #POINTER(VecFx32)),
        ('nrmDataOffset', u32), #POINTER(VecFx16)),
        ('prismDataOffset', u32), #POINTER(kcol_prism_data_t)),
        ('blockDataOffset', u32), #POINTER(u32)),
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
