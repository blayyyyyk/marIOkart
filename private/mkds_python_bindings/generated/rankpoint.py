from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.types import *


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_rankpoint_h_3_9(Structure):
    _fields_ = [
        ('points', (c_int * 8)),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_rankpoint_h_8_9(Structure):
    _fields_ = [
        ('signature', c_int),
        ('fileSize', c_int),
        ('entries', (rankpoint_entry_t * 1)),
    ]

class rankpoint_t(Structure):
    _fields_ = [
        ('signature', u32),
        ('fileSize', u32),
        ('entries', (rankpoint_entry_t * 1)),
    ]

class rankpoint_entry_t(Structure):
    _fields_ = [
        ('points', (u8 * 8)),
    ]
