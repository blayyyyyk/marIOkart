from ctypes import *
from private.mkds_python_bindings.testing.types import *


class MATHRandContext32(Structure):
    _fields_ = [
        ('x', u64),
        ('mul', u64),
        ('add', u64),
    ]

class MATHRandContext16(Structure):
    _fields_ = [
        ('x', u32),
        ('mul', u32),
        ('add', u32),
    ]
