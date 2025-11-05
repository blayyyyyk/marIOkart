from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.mapobjInstance import *
from private.mkds_python_bindings.generated.spaEmitter import *
from private.mkds_python_bindings.generated.types import *

FLIP_STATE_INIT = 0
FLIP_STATE_IDLE = 1
FLIP_STATE_FLIP_IN = 2
FLIP_STATE_FLIP_OUT = 3
FLIP_STATE_BALL_HIT_WAIT = 4
FLIP_STATE_BALL_HIT_FLIP_IN = 5
FLIP_STATE_BALL_HIT_FLIP_OUT = 6

FLIP_STATE_INIT = 0
FLIP_STATE_IDLE = 1
FLIP_STATE_FLIP_IN = 2
FLIP_STATE_FLIP_OUT = 3
FLIP_STATE_BALL_HIT_WAIT = 4
FLIP_STATE_BALL_HIT_FLIP_IN = 5
FLIP_STATE_BALL_HIT_FLIP_OUT = 6


class flipper_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('fieldA0', c_int),
        ('modelFlip', BOOL),
        ('baseMatrices', (MtxFx43 * 30)),
        ('extraColMatrices', (MtxFx43 * 30)),
        ('ptclEmitterPositions', (VecFx32 * 30)),
        ('ptclEmitterTargets', (VecFx32 * 30)),
        ('waitCounter', c_int),
        ('ballHitFrameCounter', c_int),
        ('animFrame', c_int),
        ('nsbtpFrame', u16),
        ('nsbtaFrame', u16),
        ('electricityActive', BOOL),
        ('state', FlipperState),
        ('ptclEmitter', POINTER32(spa_emitter_t)),
        ('driverHitTimeouts', (c_int * 8)),
    ]
FlipperState = c_int
