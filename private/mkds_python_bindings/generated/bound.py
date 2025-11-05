from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.mapobjInstance import *
from private.mkds_python_bindings.generated.pathwalker import *
from private.mkds_python_bindings.generated.sinThing import *
from private.mkds_python_bindings.generated.types import *

BOUND_STATE_IDLE = 0
BOUND_STATE_BOUNCING = 1

BOUND_STATE_IDLE = 0
BOUND_STATE_BOUNCING = 1


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_mapobj_enemies_bound_h_12_9(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('nsbtpFrame', u16),
        ('fieldA4', u32),
        ('fieldA8', u32),
        ('fieldAC', u32),
        ('fieldB0', u32),
        ('width', u32),
        ('scaleXZSinThing', sinthing_t),
        ('scaleYSinThing', sinthing_t),
        ('pathwalker', pw_pathwalker_t),
        ('state', BoundState),
        ('driverHitTimeouts', (u32 * 8)),
    ]
BoundState = c_int
