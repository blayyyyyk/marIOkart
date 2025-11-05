from ctypes import *
from private.mkds_python_bindings.testing.types import *

NNSFndHeapHandle = u32
s16 = c_int16
s32 = c_int
s8 = c_char
u16 = c_short
u32 = c_uint
u8 = c_char


class NNSFndLink(Structure):
    _fields_ = [
        ('prev', u32), # 4
        ('next', u32), # 4
    ]

class NNSFndList(Structure):
    _fields_ = [
        ('head', u32), # 4
        ('tail', u32), # 4
        ('numObjects', c_int), # 4
    ]

class NNSFndAllocator(Structure):
    _fields_ = [
        ('allocFunc', u32),
        ('freeFunc', u32),
        ('heapHandle', u32),
    ]

class NNSFndHeap(Structure):
    _fields_ = [
        ('base', u32),
        ('size', u32),
        ('flags', u32),
    ]

class NNSFndHeapBlockHead(Structure):
    _fields_ = [
        ('prev', u32),
        ('next', u32),
        ('size', u32),
        ('signature', u16),
        ('attributes', u16),
    ]

class NNSFndExpHeap(Structure):
    _fields_ = [
        ('heapHandle', NNSFndHeapHandle),
        ('totalSize', u32),
        ('usedSize', u32),
        ('freeSize', u32),
    ]

class NNSFndFrmHeap(Structure):
    _fields_ = [
        ('heapHandle', NNSFndHeapHandle),
        ('frameStart', u32),
        ('frameEnd', u32),
        ('currentOffset', u32),
    ]

class NNSFndMemoryRegion(Structure):
    _fields_ = [
        ('start', u32),
        ('end', u32),
    ]

class NNSFndArena(Structure):
    _fields_ = [
        ('region', NNSFndMemoryRegion),
        ('name', u32), #POINTER(c_char)),
    ]

class NNSFndArchive(Structure):
    _fields_ = [
        ('startAddr', u32),
        ('fatData', u32),
        ('fntData', u32),
        ('fileCount', u32),
    ]

class NNSFndSimpleAllocator(Structure):
    _fields_ = [
        ('allocFunc', u32),
        ('freeFunc', u32),
        ('heapHandle', u32),
    ]
