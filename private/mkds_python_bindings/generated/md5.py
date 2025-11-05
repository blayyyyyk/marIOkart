from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
UINT2 = c_short
POINTER = c_void_p32

class MD5_CTX(Structure):
    _fields_ = [
        ('state', (UINT4 * 4)),
        ('count', (UINT4 * 2)),
        ('buffer', (c_char * 64)),
    ]
UINT4 = c_int
