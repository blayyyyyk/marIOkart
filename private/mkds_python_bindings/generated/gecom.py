from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.types import *


class NNSG3dGeBuffer(Structure):
    _fields_ = [
        ('idx', u32),
        ('data', (u32 * 192)),
    ]
