from ctypes import *
from private.mkds_python_bindings.testing.pathwalker import *
from private.mkds_python_bindings.testing.sinThing import *
from private.mkds_python_bindings.testing.types import *

BOUND_STATE_IDLE = 0
BOUND_STATE_BOUNCING = 1

BOUND_STATE_IDLE = 0
BOUND_STATE_BOUNCING = 1

BoundState = c_int


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_race_mapobj_enemies_bound_h_11_9_(Structure):
    _fields_ = [
        ('mobj', c_int),
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
        ('driverHitTimeouts', u32),
    ]
