from ctypes import *
from private.mkds_python_bindings.testing.fx import *


class quaternion_t(Structure):
    _fields_ = [
        ('x', fx32),
        ('y', fx32),
        ('z', fx32),
        ('w', fx32),
    ]
