from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.mapobjInstance import *
from private.mkds_python_bindings.generated.pathwalker import *
from private.mkds_python_bindings.generated.sinThing import *
from private.mkds_python_bindings.generated.types import *

SANBO_STATE_IDLE = 0
SANBO_STATE_1 = 1
SANBO_STATE_DIEING = 2
SANBO_STATE_RESURRECTING = 3

SANBO_STATE_IDLE = 0
SANBO_STATE_1 = 1
SANBO_STATE_DIEING = 2
SANBO_STATE_RESURRECTING = 3


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_mapobj_enemies_sanbo_h_14_9(Structure):
    _fields_ = [
        ('dieingPosition', VecFx32),
        ('dieingVelocity', VecFx32),
        ('scaleXY', fx32),
        ('rotZSinThing', sinthing_t),
        ('rotZ', c_int),
        ('rotZSpeed', c_int),
        ('wiggleWaitCounter', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_mapobj_enemies_sanbo_h_25_9(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('pwWaitCounter', c_int),
        ('hitTimeout', c_int),
        ('resurrectionWaitCounter', c_int),
        ('sfxTimeout', c_int),
        ('bodyParts', (sanbo_part_t * 4)),
        ('bodyPartCount', u16),
        ('pwSpeed', fx32),
        ('pathwalker', pw_pathwalker_t),
        ('state', SanboState),
    ]
SanboState = c_int
