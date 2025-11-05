from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.mapobjInstance import *
from private.mkds_python_bindings.generated.mapobjLogicPart import *
from private.mkds_python_bindings.generated.objectShadow import *
from private.mkds_python_bindings.generated.spaEmitter import *
from private.mkds_python_bindings.generated.types import *

EFBUB_STATE_JUMP = 0
EFBUB_STATE_WAIT = 1
EFBUB_STATE_COUNT = 2

EFBUB_STATE_JUMP = 0
EFBUB_STATE_WAIT = 1
EFBUB_STATE_COUNT = 2


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_mapobj_enemies_mkdEfBubble_h_15_9(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('waitTime', c_int),
        ('lowYPos', fx32),
        ('rotation', u16),
        ('highYPos', fx32),
        ('shadow', objshadow_t),
        ('driverHitMask', u8),
        ('emitter69', POINTER32(spa_emitter_t)),
        ('emitter70', POINTER32(spa_emitter_t)),
        ('emitterFail', BOOL),
    ]

class efbub_logic_part_t(Structure):
    _fields_ = [
        ('logicPart', mobj_logic_part_t),
        ('field28', u32),
    ]
EfbubState = c_int
