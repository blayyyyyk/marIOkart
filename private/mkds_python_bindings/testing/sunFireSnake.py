from ctypes import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.mapobjInstance import *
from private.mkds_python_bindings.testing.spaEmitter import *
from private.mkds_python_bindings.testing.types import *

SFSN_STATE_0 = 0
SFSN_STATE_1 = 1
SFSN_STATE_2 = 2
SFSN_STATE_3 = 3

SFSN_STATE_0 = 0
SFSN_STATE_1 = 1
SFSN_STATE_2 = 2
SFSN_STATE_3 = 3

SunFireSnakeState = c_int


class sfsn_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('state1Counter', u16),
        ('state0Counter', s32),
        ('spawnPos', VecFx32),
        ('angle', u16),
        ('state', SunFireSnakeState),
        ('ptclEmitterA', u32), #POINTER(spa_emitter_t)),
        ('ptclEmitterB', u32), #POINTER(spa_emitter_t)),
    ]
