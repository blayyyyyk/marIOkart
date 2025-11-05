from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.mapobjInstance import *
from private.mkds_python_bindings.generated.pathwalker import *
from private.mkds_python_bindings.generated.sinThing import *

ASHIP_STATE_IDLE = 0
ASHIP_STATE_TAKING_OFF = 1
ASHIP_STATE_FLYING = 2

ASHIP_STATE_IDLE = 0
ASHIP_STATE_TAKING_OFF = 1
ASHIP_STATE_FLYING = 2


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_mapobj_scenery_airship_h_13_9(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('baseElevation', fx32),
        ('speed', fx32),
        ('fieldA8', sinthing_t),
        ('pathwalker', pw_pathwalker_t),
        ('state', AirshipState),
    ]
AirshipState = c_int
