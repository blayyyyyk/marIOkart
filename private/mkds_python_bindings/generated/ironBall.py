from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.mapobjInstance import *
from private.mkds_python_bindings.generated.objectShadow import *
from private.mkds_python_bindings.generated.pathwalker import *

IBALL_STATE_0 = 0
IBALL_STATE_1 = 1
IBALL_STATE_2 = 2

IBALL_STATE_0 = 0
IBALL_STATE_1 = 1
IBALL_STATE_2 = 2


class iball_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('rotZ', c_int),
        ('fieldA4', c_int),
        ('fieldA8', c_int),
        ('fieldAC', c_int),
        ('fieldB0', c_int),
        ('fieldB4', c_int),
        ('fieldB8', c_int),
        ('elevation', c_int),
        ('fieldC0', fx32),
        ('pathwalker', pw_pathwalker_t),
        ('shadow', objshadow_t),
        ('routePos', VecFx32),
        ('clipAreaMask', c_int),
        ('field12C', c_int),
        ('state', IballState),
    ]
IballState = c_int
