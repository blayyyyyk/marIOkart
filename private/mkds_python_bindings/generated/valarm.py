from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.alarm import *
from private.mkds_python_bindings.generated.types import *


class OSiVAlarm(Structure):
    _fields_ = [
        ('handler', OSAlarmHandler),
        ('arg', c_void_p32),
        ('tag', u32),
        ('frame', u32),
        ('fire', s16),
        ('delay', s16),
        ('prev', POINTER32(OSVAlarm)),
        ('next', POINTER32(OSVAlarm)),
        ('period', BOOL),
        ('finish', BOOL),
        ('canceled', BOOL),
    ]
OSVAlarmHandler = c_void_p32
