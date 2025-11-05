from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.types import *


class OSArgumentBuffer(Structure):
    _fields_ = [
        ('argMark', (c_char * 18)),
        ('size', u16),
        ('buffer', (c_char * 256)),
    ]
