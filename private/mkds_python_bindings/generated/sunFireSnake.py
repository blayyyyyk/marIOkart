from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.mapobjInstance import *
from private.mkds_python_bindings.generated.spaEmitter import *
from private.mkds_python_bindings.generated.types import *

SFSN_STATE_0 = 0
SFSN_STATE_1 = 1
SFSN_STATE_2 = 2
SFSN_STATE_3 = 3

SFSN_STATE_0 = 0
SFSN_STATE_1 = 1
SFSN_STATE_2 = 2
SFSN_STATE_3 = 3


class sfsn_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('state1Counter', u16),
        ('state0Counter', s32),
        ('spawnPos', VecFx32),
        ('angle', u16),
        ('state', SunFireSnakeState),
        ('ptclEmitterA', POINTER32(spa_emitter_t)),
        ('ptclEmitterB', POINTER32(spa_emitter_t)),
    ]
SunFireSnakeState = c_int
