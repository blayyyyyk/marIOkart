from ctypes import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.mapobjInstance import *
from private.mkds_python_bindings.testing.types import *

BAKUBAKU_STATE_IDLE = 0
BAKUBAKU_STATE_JUMP_OUT = 1
BAKUBAKU_STATE_JUMPING = 2
BAKUBAKU_STATE_JUMP_IN = 3
BAKUBAKU_STATE_TIMEOUT = 4

BAKUBAKU_STATE_IDLE = 0
BAKUBAKU_STATE_JUMP_OUT = 1
BAKUBAKU_STATE_JUMPING = 2
BAKUBAKU_STATE_JUMP_IN = 3
BAKUBAKU_STATE_TIMEOUT = 4

BakubakuState = c_int


class bakubaku_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('waitCounter', s32),
        ('triggerPos', VecFx32),
        ('state', u32),
        ('fieldB4', u32),
    ]
