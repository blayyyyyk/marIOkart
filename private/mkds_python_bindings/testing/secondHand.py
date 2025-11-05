from ctypes import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.mapobjInstance import *
from private.mkds_python_bindings.testing.quaternion import *
from private.mkds_python_bindings.testing.types import *

SECONDHAND_STATE_ACCELERATE = 0
SECONDHAND_STATE_MOVE = 1
SECONDHAND_STATE_DECELERATE = 2
SECONDHAND_STATE_WAIT = 3

SECONDHAND_STATE_ACCELERATE = 0
SECONDHAND_STATE_MOVE = 1
SECONDHAND_STATE_DECELERATE = 2
SECONDHAND_STATE_WAIT = 3

SecondHandState = c_int


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
