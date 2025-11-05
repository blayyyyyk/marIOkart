from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.mapobj import *
from private.mkds_python_bindings.generated.mapobjInstance import *
from private.mkds_python_bindings.generated.sinThing import *
from private.mkds_python_bindings.generated.types import *

TERESA_STATE_0 = 0
TERESA_STATE_1 = 1

TERESA_STATE_0 = 0
TERESA_STATE_1 = 1


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_mapobj_scenery_teresa_h_11_9(Structure):
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
TeresaState = c_int
