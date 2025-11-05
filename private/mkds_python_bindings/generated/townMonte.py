from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.mapobj import *
from private.mkds_python_bindings.generated.mapobjInstance import *
from private.mkds_python_bindings.generated.types import *


class townmonte_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('nsbtpFrame', u16),
        ('sfxId', u16),
        ('fieldA4', idk_struct_t),
    ]
