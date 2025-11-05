from ctypes import *
from private.mkds_python_bindings.testing.thread import *
from private.mkds_python_bindings.testing.types import *

OSMessage = u32


class OSMessageQueue(Structure):
    _fields_ = [
        ('queueSend', _OSThreadQueue),
        ('queueReceive', _OSThreadQueue),
        ('msgArray', u32), #POINTER(OSMessage)),
        ('msgCount', s32),
        ('firstIndex', s32),
        ('usedCount', s32),
    ]
