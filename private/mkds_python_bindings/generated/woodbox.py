from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.mapobjInstance import *
from private.mkds_python_bindings.generated.objectShadow import *
from private.mkds_python_bindings.generated.types import *

WBOX_STATE_IDLE = 0
WBOX_STATE_DEAD = 1
WBOX_STATE_REAPPEAR_BEGIN = 2
WBOX_STATE_REAPPEAR_END = 3
WBOX_STATE_COUNT = 4

WBOX_STATE_IDLE = 0
WBOX_STATE_DEAD = 1
WBOX_STATE_REAPPEAR_BEGIN = 2
WBOX_STATE_REAPPEAR_END = 3
WBOX_STATE_COUNT = 4


class wbox_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('fieldA0', VecFx32),
        ('fieldAC', u16),
        ('fieldB0', VecFx32),
        ('floorNormal', VecFx32),
        ('rotY', u16),
        ('fieldCC', VecFx32),
        ('fieldD8', VecFx32),
        ('fieldE4', VecFx32),
        ('fieldF0', u16),
        ('fieldF4', VecFx32),
        ('field100', VecFx32),
        ('field10C', u16),
        ('field10E', u16),
        ('field110', u16),
        ('shadow', objshadow_t),
    ]
WoodBoxState = c_int
