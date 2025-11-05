from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.types import *


class WCMConfig(Structure):
    _fields_ = [
        ('dmano', s32),
        ('pbdbuffer', c_void_p32),
        ('nbdbuffer', s32),
        ('nbdmode', s32),
    ]

class WCMNotice(Structure):
    _fields_ = [
        ('notify', s16),
        ('result', s16),
        ('parameter', (WCMNoticeParameter * 3)),
    ]

class WCMWepDesc(Structure):
    _fields_ = [
        ('mode', u8),
        ('keyId', u8),
        ('key', (u8 * 80)),
    ]
WCMNotify = c_void_p32

class WCMNoticeParameter(Union):
    _fields_ = [
        ('n', s32),
        ('p', c_void_p32),
    ]
