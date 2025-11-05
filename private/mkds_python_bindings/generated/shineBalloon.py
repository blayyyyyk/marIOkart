from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.driver import *
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.mapobjInstance import *
from private.mkds_python_bindings.generated.types import *

SBLLN_STATE_0 = 0
SBLLN_STATE_1 = 1
SBLLN_STATE_2 = 2
SBLLN_STATE_3 = 3

SBLLN_STATE_0 = 0
SBLLN_STATE_1 = 1
SBLLN_STATE_2 = 2
SBLLN_STATE_3 = 3


class sblln_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('scale', fx32),
        ('scaleDelta', fx32),
        ('counter', s32),
        ('driver', POINTER32(driver_t)),
        ('state', ShineBalloonState),
    ]
ShineBalloonState = c_int
