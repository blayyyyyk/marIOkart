from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
GPIPeer_st = c_void_p32

class GPIBuffer(Structure):
    _fields_ = [
        ('buffer', POINTER32(c_char)),
        ('size', c_int),
        ('len', c_int),
        ('pos', c_int),
    ]
