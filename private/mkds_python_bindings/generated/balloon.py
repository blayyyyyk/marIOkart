from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.mapobjInstance import *
from private.mkds_python_bindings.generated.types import *

BALLOON_STATE_0 = 0
BALLOON_STATE_1 = 1
BALLOON_STATE_2 = 2

BALLOON_STATE_0 = 0
BALLOON_STATE_1 = 1
BALLOON_STATE_2 = 2


class balloon_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('fieldA0', c_int),
        ('driverId', c_int),
        ('color', c_int),
        ('gapAC', u32),
        ('fieldB0', VecFx32),
        ('fieldBC', VecFx32),
        ('fieldC8', c_int),
        ('fieldCC', c_int),
        ('inflationProgress', c_int),
        ('inflationDelta', c_int),
        ('scale3', c_int),
        ('scale3Delta', c_int),
        ('fieldE0', c_int),
        ('scale', c_int),
        ('fieldE8', VecFx32),
        ('subBalloonCountPlusOne', c_int),
        ('subBalloons', POINTER32(POINTER32(balloon_t))),
        ('state', BalloonState),
    ]
BalloonState = c_int
