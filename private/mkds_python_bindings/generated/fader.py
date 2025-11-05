from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32

class NNSSndFader(Structure):
    _fields_ = [
        ('origin', c_int),
        ('target', c_int),
        ('counter', c_int),
        ('frame', c_int),
    ]
