from ctypes import *
from private.mkds_python_bindings.testing.mapobjInstance import *
from private.mkds_python_bindings.testing.types import *


class shinc_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('hasSpawned', BOOL),
        ('fieldA4', u32),
        ('counter', c_int),
    ]
