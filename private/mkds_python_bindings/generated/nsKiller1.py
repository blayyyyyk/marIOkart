from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.mapobjInstance import *
from private.mkds_python_bindings.generated.pathwalker import *
from private.mkds_python_bindings.generated.types import *

NSK1_STATE_WAIT = 0
NSK1_STATE_MOVE = 1
NSK1_STATE_SHOOT = 2
NSK1_STATE_EMIT = 3

NSK1_STATE_WAIT = 0
NSK1_STATE_MOVE = 1
NSK1_STATE_SHOOT = 2
NSK1_STATE_EMIT = 3


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_mapobj_enemies_nsKiller1_h_14_9(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('counter', s32),
        ('unkA4', (u8 * 4)),
        ('state', NsKiller1State),
        ('pathwalker', pw_pathwalker_t),
    ]
NsKiller1State = c_int
