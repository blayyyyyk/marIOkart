from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.player import *
from private.mkds_python_bindings.generated.types import *


class sound_pool_t(Structure):
    _fields_ = [
        ('nrElements', c_int),
        ('elements1Idx', u32),
        ('elements1', c_void_p32),
        ('elements', c_void_p32),
        ('elementSize', u32),
    ]

class sp_handle_t(Structure):
    _fields_ = [
        ('handle', NNSSndHandle),
        ('age', u32),
    ]
