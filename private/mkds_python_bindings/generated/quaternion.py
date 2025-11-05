from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.fx import *


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_math_quaternion_h_3_9(Structure):
    _fields_ = [
        ('x', c_int),
        ('y', c_int),
        ('z', c_int),
        ('w', c_int),
    ]

class quaternion_t(Structure):
    _fields_ = [
        ('x', fx32),
        ('y', fx32),
        ('z', fx32),
        ('w', fx32),
    ]
