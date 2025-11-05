from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32

class struc_333(Structure):
    _fields_ = [
        ('field0', c_int),
        ('field4', c_int),
        ('field8', c_int),
        ('fieldC', c_int),
        ('field10', c_int),
        ('field14', c_int),
        ('field18', c_int),
    ]
