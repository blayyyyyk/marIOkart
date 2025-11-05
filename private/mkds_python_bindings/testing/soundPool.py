from ctypes import *
from private.mkds_python_bindings.testing.types import *


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_snd_soundPool_h_3_9_(Structure):
    _fields_ = [
        ('handle', c_int),
        ('age', u32),
    ]

class sound_pool_t(Structure):
    _fields_ = [
        ('nrElements', c_int),
        ('elements1Idx', u32),
        ('elements1', u32),
        ('elements', u32),
        ('elementSize', u32),
    ]
