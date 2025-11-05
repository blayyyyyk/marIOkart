from ctypes import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.mapobjInstance import *
from private.mkds_python_bindings.testing.pathwalker import *
from private.mkds_python_bindings.testing.types import *

EFBNR_STATE_WAIT = 0
EFBNR_STATE_BURN_START = 1
EFBNR_STATE_BURN = 2
EFBNR_STATE_BURN_STOP = 3
EFBNR_STATE_BURN_CONTINUOUS = 4

EFBNR_STATE_WAIT = 0
EFBNR_STATE_BURN_START = 1
EFBNR_STATE_BURN = 2
EFBNR_STATE_BURN_STOP = 3
EFBNR_STATE_BURN_CONTINUOUS = 4

EfbnrState = c_int


class efbnr_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('counter', s32),
        ('nsbtaFrame', u16),
        ('scale', fx32),
        ('state', EfbnrState),
        ('pathwalker', pw_pathwalker_t),
        ('driverHitTimeouts', (s32 * 8)),
    ]
