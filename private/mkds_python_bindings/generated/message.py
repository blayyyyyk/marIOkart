from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.thread import *
from private.mkds_python_bindings.generated.types import *


class OSMessageQueue(Structure):
    _fields_ = [
        ('queueSend', OSThreadQueue),
        ('queueReceive', OSThreadQueue),
        ('msgArray', POINTER32(OSMessage)),
        ('msgCount', s32),
        ('firstIndex', s32),
        ('usedCount', s32),
    ]
OSMessage = c_void_p32
