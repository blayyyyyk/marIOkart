from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.model import *
from private.mkds_python_bindings.generated.types import *


class shadowmodel_t(Structure):
    _fields_ = [
        ('model', model_t),
        ('polygonId', u16),
        ('alpha', u16),
        ('flags', u16),
    ]
