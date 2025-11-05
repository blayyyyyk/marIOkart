from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.dynamicCollision import *
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.types import *

ROTCYL_STATE_BEGIN_ROTATE = 0
ROTCYL_STATE_ROTATE = 1
ROTCYL_STATE_END_ROTATE = 2
ROTCYL_STATE_IDLE = 3
ROTCYL_STATE_COUNT = 4

ROTCYL_STATE_BEGIN_ROTATE = 0
ROTCYL_STATE_ROTATE = 1
ROTCYL_STATE_END_ROTATE = 2
ROTCYL_STATE_IDLE = 3
ROTCYL_STATE_COUNT = 4

ROTCYL_TYPE_TEST_CYLINDER = 0
ROTCYL_TYPE_GEAR_BLACK = 1
ROTCYL_TYPE_GEAR_WHITE = 2
ROTCYL_TYPE_ROTARY_ROOM = 3
ROTCYL_TYPE_ROTARY_BRIDGE = 4

ROTCYL_TYPE_TEST_CYLINDER = 0
ROTCYL_TYPE_GEAR_BLACK = 1
ROTCYL_TYPE_GEAR_WHITE = 2
ROTCYL_TYPE_ROTARY_ROOM = 3
ROTCYL_TYPE_ROTARY_BRIDGE = 4


class rotcyl_t(Structure):
    _fields_ = [
        ('dcol', dcol_inst_t),
        ('startStopDuration', u16),
        ('rotateDuration', u16),
        ('idleDuration', u16),
        ('rotYVelocity', s16),
        ('velocityProgress', fx32),
        ('startStopSpeed', fx32),
        ('field154', u32),
        ('type', RotatingCylinderType),
        ('sfxId', u32),
    ]

class rotcyl_params_t(Structure):
    _fields_ = [
        ('isXZFloor', BOOL),
        ('sizeX', fx32),
        ('sizeY', fx32),
        ('type', RotatingCylinderType),
        ('dcolFloorThreshold', u32),
        ('dcolField138', u32),
        ('sfxId', u32),
    ]
RotatingCylinderState = c_int
RotatingCylinderType = c_int
