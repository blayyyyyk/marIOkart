from ctypes import *
from private.mkds_python_bindings.testing.fx import *


class kofs_entry_t(Structure):
    _fields_ = [
        ('tireName', (c_char * 16)),
        ('frontTireScale', fx32),
        ('tirePositions', (VecFx32 * 4)),
        ('characterPositions', (VecFx32 * 13)),
    ]

class kofs_t(Structure):
    _fields_ = [
        ('entries', (kofs_entry_t * 0)),
    ]
