from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.gsSHA1 import *

NNS_MCS_RINGBUF_DTYPE_START = 0
NNS_MCS_RINGBUF_DTYPE_MIDDLE = 1
NNS_MCS_RINGBUF_DTYPE_END = 2


class NNSiMcsRingBufferHeader(Structure):
    _fields_ = [
        ('signature', uint32_t),
        ('state', uint32_t),
        ('mrng', NNSiMcsMsgRange),
        ('brgn', NNSiMcsBufRgn),
    ]
NNSiMcsBufPtr = c_void_p32
NNSMcsRingBuffer = c_long

class NNSiMcsBufRgn(Structure):
    _fields_ = [
        ('buf', NNSiMcsUIntPtr),
        ('bufSize', uint32_t),
    ]

class NNSiMcsMsgRange(Structure):
    _fields_ = [
        ('start', NNSiMcsUIntPtr),
        ('end', NNSiMcsUIntPtr),
    ]
NNSiMcsUIntPtr = c_long
