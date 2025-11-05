from ctypes import *
from private.mkds_python_bindings.testing.types import *


class SNDBinaryFileHeader(Structure):
    _fields_ = [
        ('signature', (c_char * 4)),
        ('byteOrder', u16),
        ('version', u16),
        ('fileSize', u32),
        ('headerSize', u16),
        ('dataBlocks', u16),
    ]

class SNDBinaryBlockHeader(Structure):
    _fields_ = [
        ('kind', u32),
        ('size', u32),
    ]
