from ctypes import *
from private.mkds_python_bindings.testing.types import *

OSIrqFunction = u32


class OSIrqCallbackInfo(Structure):
    _fields_ = [
        ('func', u32),
        ('enable', u32),
        ('arg', u32),
    ]
