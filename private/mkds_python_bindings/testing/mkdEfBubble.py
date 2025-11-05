from ctypes import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.mapobjInstance import *
from private.mkds_python_bindings.testing.mapobjLogicPart import *
from private.mkds_python_bindings.testing.objectShadow import *
from private.mkds_python_bindings.testing.spaEmitter import *
from private.mkds_python_bindings.testing.types import *

EFBUB_STATE_JUMP = 0
EFBUB_STATE_WAIT = 1
EFBUB_STATE_COUNT = 2

EFBUB_STATE_JUMP = 0
EFBUB_STATE_WAIT = 1
EFBUB_STATE_COUNT = 2

EfbubState = c_int


class efbub_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('waitTime', c_int),
        ('lowYPos', fx32),
        ('rotation', u16),
        ('highYPos', fx32),
        ('shadow', objshadow_t),
        ('driverHitMask', u8),
        ('emitter69', u32), #POINTER(spa_emitter_t)),
        ('emitter70', u32), #POINTER(spa_emitter_t)),
        ('emitterFail', BOOL),
    ]

class efbub_logic_part_t(Structure):
    _fields_ = [
        ('logicPart', mobj_logic_part_t),
        ('field28', u32),
    ]
