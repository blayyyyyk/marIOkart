from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.mapobjInstance import *
from private.mkds_python_bindings.generated.sfx import *
from private.mkds_python_bindings.generated.types import *

NSK2_STATE_0 = 0
NSK2_STATE_1 = 1

NSK2_STATE_0 = 0
NSK2_STATE_1 = 1


class nsk2_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('counter', s32),
        ('state', NsKiller2State),
        ('sfxEmitterExParams', POINTER32(sfx_emitter_ex_params_t)),
    ]
NsKiller2State = c_int
