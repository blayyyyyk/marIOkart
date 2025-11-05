from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.mapobjInstance import *
from private.mkds_python_bindings.generated.pathwalker import *
from private.mkds_python_bindings.generated.types import *

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


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_mapobj_enemies_mkdEfBurner_h_15_9(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('counter', s32),
        ('nsbtaFrame', u16),
        ('scale', fx32),
        ('state', EfbnrState),
        ('pathwalker', pw_pathwalker_t),
        ('driverHitTimeouts', (s32 * 8)),
    ]
EfbnrState = c_int
