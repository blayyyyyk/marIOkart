from ctypes import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.objectShadow import *
from private.mkds_python_bindings.testing.rotDieMObj import *
from private.mkds_python_bindings.testing.types import *

PUKU_STATE_0 = 0
PUKU_STATE_1 = 1
PUKU_STATE_2 = 2
PUKU_STATE_3 = 3
PUKU_STATE_COUNT = 4

PUKU_STATE_0 = 0
PUKU_STATE_1 = 1
PUKU_STATE_2 = 2
PUKU_STATE_3 = 3
PUKU_STATE_COUNT = 4

PukupukuState = c_int


class pukupuku_t(Structure):
    _fields_ = [
        ('rdmobj', rotdiemobj_t),
        ('fieldB8', VecFx32),
        ('fieldC4', u16),
        ('shadow', objshadow_t),
        ('fieldFC', u32),
        ('field100', fx32),
        ('field104', fx32),
        ('field108', fx32),
    ]
