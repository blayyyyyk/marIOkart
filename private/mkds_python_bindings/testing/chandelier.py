from ctypes import *
from private.mkds_python_bindings.testing.mapobjInstance import *
from private.mkds_python_bindings.testing.types import *

CHND_STATE_WAIT = 0
CHND_STATE_MOVE = 1

CHND_STATE_WAIT = 0
CHND_STATE_MOVE = 1

ChandelierState = c_int


class chandelier_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('nsbcaFrame', u16),
        ('counter', s32),
        ('state', ChandelierState),
    ]
