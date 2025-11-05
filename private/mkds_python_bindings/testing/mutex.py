from ctypes import *
from private.mkds_python_bindings.testing.thread import *
from private.mkds_python_bindings.testing.types import *


class OSMutex(Structure):
    _fields_ = [
        ('queue', _OSThreadQueue),
        ('thread', u32), #POINTER(_OSThread)),
        ('count', s32),
        ('link', OSMutexLink),
    ]
