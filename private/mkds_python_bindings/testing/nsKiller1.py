from ctypes import *
from private.mkds_python_bindings.testing.mapobjInstance import *
from private.mkds_python_bindings.testing.pathwalker import *
from private.mkds_python_bindings.testing.types import *

NSK1_STATE_WAIT = 0
NSK1_STATE_MOVE = 1
NSK1_STATE_SHOOT = 2
NSK1_STATE_EMIT = 3

NSK1_STATE_WAIT = 0
NSK1_STATE_MOVE = 1
NSK1_STATE_SHOOT = 2
NSK1_STATE_EMIT = 3

NsKiller1State = c_int


class nsk1_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('counter', s32),
        ('unkA4', (u8 * 4)),
        ('state', NsKiller1State),
        ('pathwalker', pw_pathwalker_t),
    ]
