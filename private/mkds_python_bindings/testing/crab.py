from ctypes import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.mapobjInstance import *
from private.mkds_python_bindings.testing.pathwalker import *
from private.mkds_python_bindings.testing.sinThing import *
from private.mkds_python_bindings.testing.types import *

CRAB_STATE_0 = 0
CRAB_STATE_1 = 1
CRAB_STATE_2 = 2
CRAB_STATE_3 = 3

CRAB_STATE_0 = 0
CRAB_STATE_1 = 1
CRAB_STATE_2 = 2
CRAB_STATE_3 = 3

CrabState = c_int


class crab_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('fieldA0', u16),
        ('rotZ', u16),
        ('counter', c_int),
        ('bodyNsbtpFrame', fx32),
        ('handNsbtpFrame', fx32),
        ('fieldB0', sinthing_t),
        ('fieldD0', sinthing_t),
        ('pathWalker', pw_pathwalker_t),
        ('state', CrabState),
    ]
