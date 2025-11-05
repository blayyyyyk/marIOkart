from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.types import *


class NNSFndAllocator(Structure):
    _fields_ = [
        ('pFunc', POINTER32(NNSFndAllocatorFunc)),
        ('pHeap', c_void_p32),
        ('heapParam1', u32),
        ('heapParam2', u32),
    ]

class NNSFndAllocatorFunc(Structure):
    _fields_ = [
        ('pfAlloc', NNSFndFuncAllocatorAlloc),
        ('pfFree', NNSFndFuncAllocatorFree),
    ]
NNSFndFuncAllocatorFree = c_void_p32
NNSFndFuncAllocatorAlloc = c_void_p32
