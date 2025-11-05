from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_heap_h_5_9(Structure):
    _fields_ = [
        ('unknown', c_int),
        ('memoryRegionStart', c_void_p32),
        ('heapStart', c_void_p32),
        ('heapHandle', c_int),
        ('processName', POINTER32(c_char)),
    ]
NNSSndHeapDisposeCallback = c_void_p32
NNSSndHeapHandle = c_void_p32
