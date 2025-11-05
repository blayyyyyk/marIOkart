from ctypes import *
from private.mkds_python_bindings.testing.types import *


class OSLockWord(Structure):
    _fields_ = [
        ('lockFlag', u32),
        ('ownerID', u16),
        ('extension', u16),
    ]
