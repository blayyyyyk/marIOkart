from ctypes import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.mapobjInstance import *
from private.mkds_python_bindings.testing.types import *

SMAN_STATE_IDLE = 0
SMAN_STATE_HIT = 1
SMAN_STATE_FALL = 2
SMAN_STATE_RISE = 3

SMAN_STATE_IDLE = 0
SMAN_STATE_HIT = 1
SMAN_STATE_FALL = 2
SMAN_STATE_RISE = 3

SmanState = c_int


class snowman_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('headRotEnabled', BOOL),
        ('headRotZ', s32),
        ('headRotZVelocity', s32),
        ('counter', s32),
        ('headElevationProgress', fx32),
        ('bottomScale', fx32),
        ('headElevation', fx32),
        ('headElevationVelocity', fx32),
        ('headMaxElevation', fx32),
        ('headMinElevation', fx32),
        ('state', SmanState),
    ]
