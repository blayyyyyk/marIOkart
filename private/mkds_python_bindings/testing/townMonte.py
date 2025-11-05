from ctypes import *
from private.mkds_python_bindings.testing.mapobj import *
from private.mkds_python_bindings.testing.mapobjInstance import *
from private.mkds_python_bindings.testing.types import *


class townmonte_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('nsbtpFrame', u16),
        ('sfxId', u16),
        ('fieldA4', idk_struct_t),
    ]
