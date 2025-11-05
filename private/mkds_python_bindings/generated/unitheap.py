from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.types import *


class NNSiFndUntHeapHead(Structure):
    _fields_ = [
        ('mbFreeList', NNSiFndUntMBlockList),
        ('mBlkSize', u32),
    ]

class NNSiFndUntMBlockList(Structure):
    _fields_ = [
        ('head', POINTER32(NNSiFndUntHeapMBlockHead)),
    ]

class NNSiFndUntHeapMBlockHead(Structure):
    _fields_ = [
        ('pMBlkHdNext', POINTER32(NNSiFndUntHeapMBlockHead)),
    ]
