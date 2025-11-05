from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.types import *


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_raceTime_h_3_9(Structure):
    _fields_ = [
        ('milliseconds', c_int),
        ('minutes', c_int),
        ('seconds', c_int),
    ]

class race_time_t(Structure):
    _fields_ = [
        ('milliseconds', u16),
        ('minutes', u8),
        ('seconds', u8),
    ]
