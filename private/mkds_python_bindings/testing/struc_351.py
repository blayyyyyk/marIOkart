from ctypes import *
from private.mkds_python_bindings.testing.types import *


class struc_351(Structure):
    _fields_ = [
        ('posX', s16),
        ('posY', s16),
        ('posZ', s16),
        ('speed', s16),
        ('driftRotY', s16),
        ('yRot', u16),
        ('flags', u32),
        ('field514field48', c_int),
        ('field514field44', c_int),
    ]
