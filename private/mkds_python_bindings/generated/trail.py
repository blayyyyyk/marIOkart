from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.list import *
from private.mkds_python_bindings.generated.model import *
from private.mkds_python_bindings.generated.types import *


class trl_texparams_t(Structure):
    _fields_ = [
        ('texImageParam', u32),
        ('field4', (u8 * 20)),
        ('plttAddress', u32),
        ('width', fx32),
        ('height', fx32),
    ]

class trl_state_t(Structure):
    _fields_ = [
        ('activeTrailList', NNSFndList),
        ('freeTrailList', NNSFndList),
        ('freePointList', NNSFndList),
        ('texture', model_res_t),
        ('polygonAttr', u32),
    ]

class trl_trail_t(Structure):
    _fields_ = [
        ('link', NNSFndLink),
        ('pointList', NNSFndList),
        ('targetPosition', POINTER32(VecFx32)),
        ('shouldDie', BOOL),
        ('initialPointSize', fx32),
        ('driverId', u32),
        ('driverOffset', VecFx32),
    ]

class trl_point_t(Structure):
    _fields_ = [
        ('link', NNSFndLink),
        ('size', fx32),
        ('age', u16),
        ('position', VecFx32),
    ]
