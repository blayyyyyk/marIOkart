from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.types import *


class MATHRandContext16(Structure):
    _fields_ = [
        ('x', u32),
        ('mul', u32),
        ('add', u32),
    ]

class MATHRandContext32(Structure):
    _fields_ = [
        ('x', u64),
        ('mul', u64),
        ('add', u64),
    ]
