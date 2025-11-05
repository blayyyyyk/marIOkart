from ctypes import *
from private.mkds_python_bindings.testing.types import *

MI_COMPRESSION_LZ = 16
MI_COMPRESSION_HUFFMAN = 32
MI_COMPRESSION_RL = 48
MI_COMPRESSION_DIFF = 128
MI_COMPRESSION_TYPE_MASK = 240
MI_COMPRESSION_TYPE_EX_MASK = 255

MI_COMPRESSION_LZ = 16
MI_COMPRESSION_HUFFMAN = 32
MI_COMPRESSION_RL = 48
MI_COMPRESSION_DIFF = 128
MI_COMPRESSION_TYPE_MASK = 240
MI_COMPRESSION_TYPE_EX_MASK = 255

MICompressionType = c_int


class MICompressionHeader(Structure):
    _fields_ = [
        ('compParam', u32),
        ('compType', u32),
        ('destSize', u32),
    ]

class MIUnpackBitsParam(Structure):
    _fields_ = [
        ('srcNum', u16),
        ('srcBitNum', u16),
        ('destBitNum', u16),
        ('destOffset', u32),
        ('destOffset0_on', u32),
    ]
