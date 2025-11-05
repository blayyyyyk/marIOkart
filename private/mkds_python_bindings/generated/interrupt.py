from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.types import *


class OSIrqCallbackInfo(Structure):
    _fields_ = [
        ('func', c_void_p32),
        ('enable', u32),
        ('arg', c_void_p32),
    ]
OSIrqFunction = c_void_p32
