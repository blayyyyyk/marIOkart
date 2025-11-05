from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.mapobjInstance import *
from private.mkds_python_bindings.generated.types import *

PICT_TYPE_1 = 0
PICT_TYPE_2 = 1

PICT_TYPE_1 = 0
PICT_TYPE_2 = 1

PICT_STATE_IDLE = 0
PICT_STATE_WAIT = 1
PICT_STATE_MOVE = 2

PICT_STATE_IDLE = 0
PICT_STATE_WAIT = 1
PICT_STATE_MOVE = 2


class picture_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('nsbcaFrame', u16),
        ('pictureType', PictType),
        ('counter', s32),
        ('state', PictState),
    ]
PictType = c_int
PictState = c_int
