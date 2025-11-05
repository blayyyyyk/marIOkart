from ctypes import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.mapobj import *
from private.mkds_python_bindings.testing.mapobjInstance import *
from private.mkds_python_bindings.testing.sinThing import *
from private.mkds_python_bindings.testing.types import *

TERESA_STATE_0 = 0
TERESA_STATE_1 = 1

TERESA_STATE_0 = 0
TERESA_STATE_1 = 1

TeresaState = c_int


class teresa_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('fieldA0', VecFx32),
        ('alpha', u16),
        ('nsbtpFrame', u16),
        ('flip', u16),
        ('state1Counter', s32),
        ('fieldB8', s32),
        ('fieldBC', sinthing_t),
        ('fieldDC', sinthing_t),
        ('fieldFC', idk_struct2_t),
        ('field10C', u32),
        ('state', TeresaState),
    ]
