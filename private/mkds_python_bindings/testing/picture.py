from ctypes import *
from private.mkds_python_bindings.testing.mapobjInstance import *
from private.mkds_python_bindings.testing.types import *

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

PictState = c_int
PictType = c_int


class picture_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('nsbcaFrame', u16),
        ('pictureType', PictType),
        ('counter', s32),
        ('state', PictState),
    ]
