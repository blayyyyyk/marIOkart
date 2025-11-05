from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.data import *
from private.mkds_python_bindings.generated.types import *


class NNSSndSeqArc(Structure):
    _fields_ = [
        ('fileHeader', SNDBinaryFileHeader),
        ('blockHeader', SNDBinaryBlockHeader),
        ('baseOffset', u32),
        ('count', u32),
        ('info', (NNSSndSeqArcSeqInfo * 0)),
    ]

class NNSSndSeqData(Structure):
    _fields_ = [
        ('fileHeader', SNDBinaryFileHeader),
        ('blockHeader', SNDBinaryBlockHeader),
        ('baseOffset', u32),
        ('data', (u32 * 0)),
    ]

class NNSSndSeqArcSeqInfo(Structure):
    _fields_ = [
        ('offset', u32),
        ('param', NNSSndSeqParam),
    ]

class NNSSndSeqParam(Structure):
    _fields_ = [
        ('bankNo', u16),
        ('volume', u8),
        ('channelPrio', u8),
        ('playerPrio', u8),
        ('playerNo', u8),
        ('reserved', u16),
    ]
