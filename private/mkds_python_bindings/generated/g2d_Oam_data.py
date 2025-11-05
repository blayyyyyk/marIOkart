from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.types import *


class objSize(Structure):
    _fields_ = [
        ('x', u16),
        ('y', u16),
    ]
