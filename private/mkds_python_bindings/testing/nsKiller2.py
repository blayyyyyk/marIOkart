from ctypes import *
from private.mkds_python_bindings.testing.mapobjInstance import *
from private.mkds_python_bindings.testing.sfx import *
from private.mkds_python_bindings.testing.types import *

NSK2_STATE_0 = 0
NSK2_STATE_1 = 1

NSK2_STATE_0 = 0
NSK2_STATE_1 = 1

NsKiller2State = c_int


class nsk2_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('counter', s32),
        ('state', NsKiller2State),
        ('sfxEmitterExParams', u32), #POINTER(sfx_emitter_ex_params_t)),
    ]
