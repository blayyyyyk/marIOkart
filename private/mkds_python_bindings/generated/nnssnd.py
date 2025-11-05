from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.types import *

NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p
NNSSndHeapHandle = c_void_p


class NNSSndHandle(Structure):
    _fields_ = [
        ('pPlayer', c_void_p32),
        ('pHeap', c_void_p32),
        ('pWave', c_void_p32),
        ('flag', u32),
    ]

class NNSSndArc(Structure):
    _fields_ = [
        ('pFileImage', c_void_p32),
        ('pHeader', c_void_p32),
        ('pSymbolTable', c_void_p32),
        ('pInfoBlock', c_void_p32),
        ('pFileBlock', c_void_p32),
        ('flag', c_int),
    ]

class NNSSndHeap(Structure):
    _fields_ = [
        ('start', c_void_p32),
        ('end', c_void_p32),
        ('current', c_void_p32),
        ('size', u32),
        ('used', u32),
        ('allocCount', u32),
        ('tag', u32),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nnssnd_h_17_16_(Structure):
    _fields_ = [
        ('start', c_void_p32),
        ('end', c_void_p32),
        ('current', c_void_p32),
        ('size', u32),
        ('used', u32),
        ('allocCount', u32),
        ('tag', u32),
    ]
