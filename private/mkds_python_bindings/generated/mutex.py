from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.thread import *
from private.mkds_python_bindings.generated.types import *


class OSMutex(Structure):
    _fields_ = [
        ('queue', OSThreadQueue),
        ('thread', POINTER32(OSThread)),
        ('count', s32),
        ('link', OSMutexLink),
    ]
