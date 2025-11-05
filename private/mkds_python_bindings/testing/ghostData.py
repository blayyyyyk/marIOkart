from ctypes import *
from private.mkds_python_bindings.testing.types import *


class ghost_time_t(Structure):
    _fields_ = [
        ('field0', u8),
        ('field1', u8),
        ('field2', u8),
    ]

class ghost_header_t(Structure):
    _fields_ = [
        ('signature', u32),
        ('character', u32),
        ('kart', u32),
        ('course', u32),
        ('_', u32),
        ('isValid', u32),
        ('flagsBit1', u32),
        ('flagsBit2_3', u32),
        ('_', u32),
        ('minutes', u32),
        ('seconds', u32),
        ('milliseconds', u32),
        ('nickname', (u16 * 10)),
        ('lapTimes', (ghost_time_t * 5)),
        ('field2F', u8),
    ]

class ghost_header_ex_t(Structure):
    _fields_ = [
        ('header', ghost_header_t),
        ('emblem', (u8 * 512)),
    ]
