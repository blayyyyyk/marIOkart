from ctypes import *
from private.mkds_python_bindings.testing.types import *


class sound_var_t(Structure):
    _fields_ = [
        ('value', s16),
        ('field2', s16),
        ('field4', s32),
        ('id', s8),
    ]
