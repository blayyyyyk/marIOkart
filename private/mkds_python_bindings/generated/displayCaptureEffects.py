from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
DCE_MODE_NONE = 0
DCE_MODE_DOUBLE_3D = 1
DCE_MODE_BLUR = 2

DCE_MODE_NONE = 0
DCE_MODE_DOUBLE_3D = 1
DCE_MODE_BLUR = 2


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_effects_displayCaptureEffects_h_15_9(Structure):
    _fields_ = [
        ('mode', DceMode),
        ('oamBuf', POINTER32(c_int)),
        ('flags', c_int),
        ('blurAmount', c_int),
        ('blurProgress', c_int),
    ]
DceMode = c_int
