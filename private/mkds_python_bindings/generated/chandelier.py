from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.mapobjInstance import *
from private.mkds_python_bindings.generated.types import *

CHND_STATE_WAIT = 0
CHND_STATE_MOVE = 1

CHND_STATE_WAIT = 0
CHND_STATE_MOVE = 1


class chandelier_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('nsbcaFrame', u16),
        ('counter', s32),
        ('state', ChandelierState),
    ]
ChandelierState = c_int
