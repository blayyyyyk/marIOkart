from ctypes import *
from private.mkds_python_bindings.testing.types import *


class OSArgumentBuffer(Structure):
    _fields_ = [
        ('argMark', (c_char * 18)),
        ('size', u16),
        ('buffer', (c_char * 256)),
    ]
