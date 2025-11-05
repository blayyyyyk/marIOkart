from ctypes import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.mapobjInstance import *
from private.mkds_python_bindings.testing.pathwalker import *
from private.mkds_python_bindings.testing.sinThing import *
from private.mkds_python_bindings.testing.types import *

SANBO_STATE_IDLE = 0
SANBO_STATE_1 = 1
SANBO_STATE_DIEING = 2
SANBO_STATE_RESURRECTING = 3

SANBO_STATE_IDLE = 0
SANBO_STATE_1 = 1
SANBO_STATE_DIEING = 2
SANBO_STATE_RESURRECTING = 3

SanboState = c_int


class sanbo_part_t(Structure):
    _fields_ = [
        ('dieingPosition', VecFx32),
        ('dieingVelocity', VecFx32),
        ('scaleXY', fx32),
        ('rotZSinThing', sinthing_t),
        ('rotZ', c_int),
        ('rotZSpeed', c_int),
        ('wiggleWaitCounter', c_int),
    ]

class sanbo_t(Structure):
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
