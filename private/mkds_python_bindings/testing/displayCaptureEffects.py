from ctypes import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.types import *

DCE_MODE_NONE = 0
DCE_MODE_DOUBLE_3D = 1
DCE_MODE_BLUR = 2

DCE_MODE_NONE = 0
DCE_MODE_DOUBLE_3D = 1
DCE_MODE_BLUR = 2

DceMode = c_int


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_race_effects_displayCaptureEffects_h_15_9_(Structure):
    _fields_ = [
        ('mode', DceMode),
        ('oamBuf', u32), #POINTER(c_int)),
        ('flags', u16),
        ('blurAmount', u8),
        ('blurProgress', fx32),
    ]
