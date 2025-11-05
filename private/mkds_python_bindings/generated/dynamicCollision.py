from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.mapobjInstance import *
from private.mkds_python_bindings.generated.model import *
from private.mkds_python_bindings.generated.types import *

DCOL_SHAPE_BOX = 0
DCOL_SHAPE_CYLINDER = 1

DCOL_SHAPE_BOX = 0
DCOL_SHAPE_CYLINDER = 1

DCOL_SHAPE_BOX = 0
DCOL_SHAPE_CYLINDER = 1

DCOL_SHAPE_BOX = 0
DCOL_SHAPE_CYLINDER = 1

DCOL_SHAPE_BOX = 0
DCOL_SHAPE_CYLINDER = 1

DCOL_SHAPE_BOX = 0
DCOL_SHAPE_CYLINDER = 1

DCOL_SHAPE_BOX = 0
DCOL_SHAPE_CYLINDER = 1

DCOL_SHAPE_BOX = 0
DCOL_SHAPE_CYLINDER = 1

DCOL_SHAPE_BOX = 0
DCOL_SHAPE_CYLINDER = 1

DCOL_SHAPE_BOX = 0
DCOL_SHAPE_CYLINDER = 1

DCOL_SHAPE_BOX = 0
DCOL_SHAPE_CYLINDER = 1

DCOL_SHAPE_BOX = 0
DCOL_SHAPE_CYLINDER = 1

dcol_render_func_t = c_void_p32

class dcol_inst_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('lastMtx', MtxFx33),
        ('baseMtx', MtxFx33),
        ('lastPosition', VecFx32),
        ('basePos', VecFx32),
        ('size', VecFx32),
        ('sizeZ2', fx32),
        ('isFloorYZ', BOOL),
        ('isFloorXZ', BOOL),
        ('isFloorXY', BOOL),
        ('isBoostPanel', BOOL),
        ('floorThreshold', fx32),
        ('field124', VecFx32),
        ('field130', u32),
        ('shape', DColShape),
        ('field138', u32),
        ('field13C', u32),
        ('model', POINTER32(model_t)),
    ]
DColShape = c_int
