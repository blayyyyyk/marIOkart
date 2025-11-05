from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.types import *


class NNSG2dBinaryFileHeader(Structure):
    _fields_ = [
        ('signature', u32),
        ('byteOrder', u16),
        ('version', u16),
        ('fileSize', u32),
        ('headerSize', u16),
        ('dataBlocks', u16),
    ]

class NNSG2dUserExDataBlock(Structure):
    _fields_ = [
        ('blkTypeID', u32),
        ('blkSize', u32),
    ]

class NNSG2dBinaryBlockHeader(Structure):
    _fields_ = [
        ('kind', u32),
        ('size', u32),
    ]
