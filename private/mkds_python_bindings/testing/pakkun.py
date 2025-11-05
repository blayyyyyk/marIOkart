from ctypes import *
from private.mkds_python_bindings.testing.mapobjInstance import *
from private.mkds_python_bindings.testing.pathwalker import *
from private.mkds_python_bindings.testing.types import *

PAKKUN_STATE_0 = 0
PAKKUN_STATE_1 = 1
PAKKUN_STATE_2 = 2
PAKKUN_STATE_3 = 3
PAKKUN_STATE_4 = 4
PAKKUN_STATE_5 = 5
PAKKUN_STATE_6 = 6

PAKKUN_STATE_0 = 0
PAKKUN_STATE_1 = 1
PAKKUN_STATE_2 = 2
PAKKUN_STATE_3 = 3
PAKKUN_STATE_4 = 4
PAKKUN_STATE_5 = 5
PAKKUN_STATE_6 = 6

PakkunState = c_int


class pakkun_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('polygonId', u16),
        ('nsbcaFrame', c_int),
        ('fieldA8', c_int),
        ('fieldAC', c_int),
        ('fieldB0', c_int),
        ('state', PakkunState),
        ('pathwalkers', (pw_pathwalker_t * 7)),
        ('counter', s32),
        ('curPath', u16),
        ('pathCount', u16),
        ('field1BC', (c_int * 7)),
        ('field1D8', c_int),
        ('mouthRotY', c_int),
        ('mouthRotX', c_int),
        ('field1E4', c_int),
        ('field1E8', c_int),
        ('field1EC', c_int),
        ('scale', c_int),
        ('scaleVelocity', c_int),
        ('headElevation', c_int),
        ('fireballElevation', c_int),
    ]
