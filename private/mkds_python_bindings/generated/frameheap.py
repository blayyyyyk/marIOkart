from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.types import *


class NNSiFndFrmHeapHead(Structure):
    _fields_ = [
        ('headAllocator', c_void_p32),
        ('tailAllocator', c_void_p32),
        ('pState', POINTER32(NNSiFndFrmHeapState)),
    ]

class NNSiFndFrmHeapState(Structure):
    _fields_ = [
        ('tagName', u32),
        ('headAllocator', c_void_p32),
        ('tailAllocator', c_void_p32),
        ('pPrevState', POINTER32(NNSiFndFrmHeapState)),
    ]
