from ctypes import *
from private.mkds_python_bindings.testing.types import *


class race_time_t(Structure):
    _fields_ = [
        ('milliseconds', u16),
        ('minutes', u8),
        ('seconds', u8),
    ]
