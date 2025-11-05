from ctypes import *
from private.mkds_python_bindings.testing.types import *


class rankpoint_entry_t(Structure):
    _fields_ = [
        ('points', (u8 * 8)),
    ]

class rankpoint_t(Structure):
    _fields_ = [
        ('signature', u32),
        ('fileSize', u32),
        ('entries', (rankpoint_entry_t * 1)),
    ]
