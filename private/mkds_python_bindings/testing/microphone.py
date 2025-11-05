from ctypes import *
from private.mkds_python_bindings.testing.types import *


class mic_t(Structure):
    _fields_ = [
        ('sampleBuffer', (s8 * 1024)),
        ('autoParam', c_int),
        ('frameCounter', c_int),
    ]
