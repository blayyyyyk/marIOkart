from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32

class _gs_crypt_key(Structure):
    _fields_ = [
        ('state', (c_char * 256)),
        ('x', c_char),
        ('y', c_char),
    ]
