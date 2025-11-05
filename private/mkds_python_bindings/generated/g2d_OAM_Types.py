from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.types import *


class NNSG2dOamAffineParams(Structure):
    _fields_ = [
        ('PA', s16),
        ('PB', s16),
        ('PC', s16),
        ('PD', s16),
    ]

class NNSG2dOamDataChunk(Structure):
    _fields_ = [
        ('attr0', u16),
        ('attr1', u16),
        ('attr2', u16),
    ]
