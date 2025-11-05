from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.mapobj import *
from private.mkds_python_bindings.generated.mapobjInstance import *
from private.mkds_python_bindings.generated.types import *

BASABASA_STATE_0 = 0
BASABASA_STATE_1 = 1
BASABASA_STATE_2 = 2
BASABASA_STATE_3 = 3

BASABASA_STATE_0 = 0
BASABASA_STATE_1 = 1
BASABASA_STATE_2 = 2
BASABASA_STATE_3 = 3


class basabasa_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('velocity', VecFx32),
        ('nsbtpFrame', u16),
        ('initialCounter', s32),
        ('state0Counter', s32),
        ('state2Counter', s32),
        ('emitSound', BOOL),
        ('mapIconType', u16),
        ('rotZ', idk_struct_t),
        ('driverHitMask', u8),
        ('state', BasabasaState),
    ]
BasabasaState = c_int
