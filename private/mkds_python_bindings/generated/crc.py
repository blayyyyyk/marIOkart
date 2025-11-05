from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.types import *


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
MATHCRC16Context = c_short
MATHCRC8Context = c_char
MATHCRC32Context = c_long
