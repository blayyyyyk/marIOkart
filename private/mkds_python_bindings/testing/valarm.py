from ctypes import *
from private.mkds_python_bindings.testing.alarm import *
from private.mkds_python_bindings.testing.types import *

OSVAlarmHandler = u32


class OSiVAlarm(Structure):
    _fields_ = [
        ('handler', OSAlarmHandler),
        ('arg', u32),
        ('tag', u32),
        ('frame', u32),
        ('fire', s16),
        ('delay', s16),
        ('prev', u32), #POINTER(OSVAlarm)),
        ('next', u32), #POINTER(OSVAlarm)),
        ('period', BOOL),
        ('finish', BOOL),
        ('canceled', BOOL),
    ]
