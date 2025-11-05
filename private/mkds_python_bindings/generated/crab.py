from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.mapobjInstance import *
from private.mkds_python_bindings.generated.pathwalker import *
from private.mkds_python_bindings.generated.sinThing import *
from private.mkds_python_bindings.generated.types import *

CRAB_STATE_0 = 0
CRAB_STATE_1 = 1
CRAB_STATE_2 = 2
CRAB_STATE_3 = 3

CRAB_STATE_0 = 0
CRAB_STATE_1 = 1
CRAB_STATE_2 = 2
CRAB_STATE_3 = 3


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_mapobj_enemies_crab_h_15_9(Structure):
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
CrabState = c_int
