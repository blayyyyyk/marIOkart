from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32

class GPIKey(Structure):
    _fields_ = [
        ('keyName', POINTER32(c_char)),
        ('keyValue', POINTER32(c_char)),
    ]
