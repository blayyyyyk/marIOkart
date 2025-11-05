from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.mapobjInstance import *
from private.mkds_python_bindings.generated.objectShadow import *
from private.mkds_python_bindings.generated.types import *

CHORO_STATE_0 = 0
CHORO_STATE_1 = 1
CHORO_STATE_2 = 2
CHORO_STATE_3 = 3
CHORO_STATE_4 = 4
CHORO_STATE_5 = 5
CHORO_STATE_6 = 6
CHORO_STATE_7 = 7

CHORO_STATE_0 = 0
CHORO_STATE_1 = 1
CHORO_STATE_2 = 2
CHORO_STATE_3 = 3
CHORO_STATE_4 = 4
CHORO_STATE_5 = 5
CHORO_STATE_6 = 6
CHORO_STATE_7 = 7


class choropu_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('setting1', u32),
        ('setting0', u32),
        ('fieldA8', u32),
        ('fieldAC', c_int),
        ('fieldB0', c_int),
        ('fieldB4', BOOL),
        ('rotZ', u16),
        ('shadow', objshadow_t),
    ]
ChoropuState = c_int
