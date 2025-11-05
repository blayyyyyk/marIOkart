from ctypes import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.mapobjInstance import *
from private.mkds_python_bindings.testing.spaEmitter import *
from private.mkds_python_bindings.testing.types import *

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

FlipperState = c_int


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
        ('ptclEmitter', u32), #POINTER(spa_emitter_t)),
        ('driverHitTimeouts', (c_int * 8)),
    ]
