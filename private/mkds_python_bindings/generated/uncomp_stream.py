from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.types import *


class MIUncompContextRL(Structure):
    _fields_ = [
        ('destp', POINTER32(u8)),
        ('destCount', s32),
        ('destTmp', u16),
        ('destTmpCnt', u8),
        ('flags', u8),
        ('length', u16),
        ('_padding', (u8 * 2)),
    ]

class MIUncompContextLZ(Structure):
    _fields_ = [
        ('destp', POINTER32(u8)),
        ('destCount', s32),
        ('destTmp', u16),
        ('destTmpCnt', u8),
        ('flags', u8),
        ('flagIndex', u8),
        ('length', u8),
        ('lengthFlg', u8),
        ('_padding', (u8 * 1)),
    ]

class MIUncompContextHuffman(Structure):
    _fields_ = [
        ('destp', POINTER32(u8)),
        ('destCount', s32),
        ('treep', POINTER32(u8)),
        ('srcTmp', u32),
        ('destTmp', u32),
        ('treeSize', s16),
        ('srcTmpCnt', u8),
        ('destTmpCnt', u8),
        ('bitSize', u8),
        ('_padding2', (u8 * 3)),
        ('tree', (u8 * 512)),
    ]
