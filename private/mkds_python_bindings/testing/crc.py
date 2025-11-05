from ctypes import *
from private.mkds_python_bindings.testing.types import *

MATHCRC16Context = c_short
MATHCRC32Context = c_int
MATHCRC8Context = c_char


class MATHCRC8Table(Structure):
    _fields_ = [
        ('table', (u8 * 256)),
    ]

class MATHCRC16Table(Structure):
    _fields_ = [
        ('table', (u16 * 256)),
    ]

class MATHCRC32Table(Structure):
    _fields_ = [
        ('table', (u32 * 256)),
    ]
