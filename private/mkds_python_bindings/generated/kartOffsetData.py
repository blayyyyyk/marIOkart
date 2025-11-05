from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.fx import *


class kofs_t(Structure):
    _fields_ = [
        ('entries', (kofs_entry_t * 0)),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_drivers_kartOffsetData_h_3_9(Structure):
    _fields_ = [
        ('tireName', (c_char * 16)),
        ('frontTireScale', c_int),
        ('tirePositions', (c_int * 4)),
        ('characterPositions', (c_int * 13)),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_drivers_kartOffsetData_h_13_9(Structure):
    _fields_ = [
        ('entries', (kofs_entry_t * 0)),
    ]

class kofs_entry_t(Structure):
    _fields_ = [
        ('tireName', (c_char * 16)),
        ('frontTireScale', fx32),
        ('tirePositions', (VecFx32 * 4)),
        ('characterPositions', (VecFx32 * 13)),
    ]
