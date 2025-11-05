from ctypes import *
from private.mkds_python_bindings.testing.types import *


class EXTKeys(Structure):
    _fields_ = [
        ('key', u16),
        ('count', u16),
    ]
