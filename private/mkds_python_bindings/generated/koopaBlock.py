from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.dynamicCollision import *
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.pathwalker import *
from private.mkds_python_bindings.generated.types import *

KOOPABLOCK_STATE_SPEEDUP = 0
KOOPABLOCK_STATE_MOVE = 1
KOOPABLOCK_STATE_SLOWDOWN = 2

KOOPABLOCK_STATE_SPEEDUP = 0
KOOPABLOCK_STATE_MOVE = 1
KOOPABLOCK_STATE_SLOWDOWN = 2


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_mapobj_obstacles_koopaBlock_h_13_9(Structure):
    _fields_ = [
        ('dcolMObj', dcol_inst_t),
        ('pathWalker', pw_pathwalker_t),
        ('speed', fx32),
        ('waitCounter', u16),
    ]
KoopaBlockState = c_int
