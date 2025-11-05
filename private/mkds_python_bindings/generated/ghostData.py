from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_drivers_ghostData_h_3_9(Structure):
    _fields_ = [
        ('field0', c_int),
        ('field1', c_int),
        ('field2', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_drivers_ghostData_h_10_9(Structure):
    _fields_ = [
        ('signature', c_int),
        ('character', c_int),
        ('kart', c_int),
        ('course', c_int),
        ('u32', c_int),
        ('isValid', c_int),
        ('flagsBit1', c_int),
        ('flagsBit2_3', c_int),
        ('minutes', c_int),
        ('seconds', c_int),
        ('milliseconds', c_int),
        ('nickname', (c_int * 10)),
        ('lapTimes', (ghost_time_t * 5)),
        ('field2F', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_drivers_ghostData_h_32_9(Structure):
    _fields_ = [
        ('header', ghost_header_t),
        ('emblem', (c_int * 512)),
    ]
