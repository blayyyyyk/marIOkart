from ctypes import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.mapobjInstance import *
from private.mkds_python_bindings.testing.model import *
from private.mkds_python_bindings.testing.types import *

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

DColShape = c_int
DColShape = c_int
DColShape = c_int
DColShape = c_int
DColShape = c_int
DColShape = c_int
dcol_render_func_t = u32
dcol_render_func_t = u32
dcol_render_func_t = u32
dcol_render_func_t = u32
dcol_render_func_t = u32
dcol_render_func_t = u32


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
        ('model', u32), #POINTER(model_t)),
    ]
