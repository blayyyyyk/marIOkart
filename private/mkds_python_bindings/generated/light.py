from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.gxcommon import *
from private.mkds_python_bindings.generated.types import *


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_light_h_3_9(Structure):
    _fields_ = [
        ('color', c_int),
        ('r', c_int),
        ('g', c_int),
        ('b', c_int),
        ('rDelta', c_int),
        ('gDelta', c_int),
        ('bDelta', c_int),
        ('lightMask', c_int),
        ('progress', c_int),
    ]

class light_t(Structure):
    _fields_ = [
        ('color', GXRgb),
        ('r', s16),
        ('g', s16),
        ('b', s16),
        ('rDelta', s16),
        ('gDelta', s16),
        ('bDelta', s16),
        ('lightMask', u16),
        ('progress', fx16),
    ]
