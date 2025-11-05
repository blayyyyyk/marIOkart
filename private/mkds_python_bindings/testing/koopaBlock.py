from ctypes import *
from private.mkds_python_bindings.testing.dynamicCollision import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.pathwalker import *
from private.mkds_python_bindings.testing.types import *

KOOPABLOCK_STATE_SPEEDUP = 0
KOOPABLOCK_STATE_MOVE = 1
KOOPABLOCK_STATE_SLOWDOWN = 2

KOOPABLOCK_STATE_SPEEDUP = 0
KOOPABLOCK_STATE_MOVE = 1
KOOPABLOCK_STATE_SLOWDOWN = 2

KoopaBlockState = c_int


class koopablock_t(Structure):
    _fields_ = [
        ('dcolMObj', dcol_inst_t),
        ('pathWalker', pw_pathwalker_t),
        ('speed', fx32),
        ('waitCounter', u16),
    ]
