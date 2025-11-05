from ctypes import *
from private.mkds_python_bindings.testing.driver import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.mapobjInstance import *
from private.mkds_python_bindings.testing.types import *

SBLLN_STATE_0 = 0
SBLLN_STATE_1 = 1
SBLLN_STATE_2 = 2
SBLLN_STATE_3 = 3

SBLLN_STATE_0 = 0
SBLLN_STATE_1 = 1
SBLLN_STATE_2 = 2
SBLLN_STATE_3 = 3

ShineBalloonState = c_int


class sblln_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('scale', fx32),
        ('scaleDelta', fx32),
        ('counter', s32),
        ('driver', u32), #POINTER(driver_t)),
        ('state', ShineBalloonState),
    ]
