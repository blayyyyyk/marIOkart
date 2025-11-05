from ctypes import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.mapobjInstance import *
from private.mkds_python_bindings.testing.pathwalker import *
from private.mkds_python_bindings.testing.sinThing import *

ASHIP_STATE_IDLE = 0
ASHIP_STATE_TAKING_OFF = 1
ASHIP_STATE_FLYING = 2

ASHIP_STATE_IDLE = 0
ASHIP_STATE_TAKING_OFF = 1
ASHIP_STATE_FLYING = 2

AirshipState = c_int


class airship_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('baseElevation', fx32),
        ('speed', fx32),
        ('fieldA8', sinthing_t),
        ('pathwalker', pw_pathwalker_t),
        ('state', AirshipState),
    ]
