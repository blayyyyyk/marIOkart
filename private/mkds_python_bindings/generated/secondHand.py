from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.mapobjInstance import *
from private.mkds_python_bindings.generated.quaternion import *
from private.mkds_python_bindings.generated.types import *

SECONDHAND_STATE_ACCELERATE = 0
SECONDHAND_STATE_MOVE = 1
SECONDHAND_STATE_DECELERATE = 2
SECONDHAND_STATE_WAIT = 3

SECONDHAND_STATE_ACCELERATE = 0
SECONDHAND_STATE_MOVE = 1
SECONDHAND_STATE_DECELERATE = 2
SECONDHAND_STATE_WAIT = 3


class secondhand_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('curRotation', quaternion_t),
        ('baseRotation', quaternion_t),
        ('startStopFrameCount', u16),
        ('oneDirFrameCount', u16),
        ('waitFrameCount', u16),
        ('baseVelocity', s16),
        ('velocity', fx32),
        ('acceleration', fx32),
    ]
SecondHandState = c_int
