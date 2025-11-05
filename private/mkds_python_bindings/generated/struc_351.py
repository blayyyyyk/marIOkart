from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.types import *


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_struc_351_h_3_9(Structure):
    _fields_ = [
        ('posX', c_int),
        ('posY', c_int),
        ('posZ', c_int),
        ('speed', c_int),
        ('driftRotY', c_int),
        ('yRot', c_int),
        ('flags', c_int),
        ('field514field48', c_int),
        ('field514field44', c_int),
    ]

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
