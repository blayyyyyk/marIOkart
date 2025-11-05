from ctypes import *
from private.mkds_python_bindings.testing.nnsfnd import *
from private.mkds_python_bindings.testing.types import *


class heap_info_t(Structure):
    _fields_ = [
        ('unknown', u32),
        ('memoryRegionStart', u32),
        ('heapStart', u32),
        ('heapHandle', NNSFndHeapHandle),
        ('processName', u32), #POINTER(c_char)),
    ]
