from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32

class RC4Context(Structure):
    _fields_ = [
        ('x', c_char),
        ('y', c_char),
        ('state', (c_char * 256)),
    ]
