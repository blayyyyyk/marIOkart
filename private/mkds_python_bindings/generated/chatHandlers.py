from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.gsPlatform import *


class ciServerMessageType(Structure):
    _fields_ = [
        ('command', POINTER32(c_char)),
        ('handler', c_void_p32),
    ]
ciCommands = c_void_p32

class ciServerMessageFilter(Structure):
    _fields_ = [
        ('type', c_int),
        ('timeout', gsi_time),
        ('name', POINTER32(c_char)),
        ('name2', POINTER32(c_char)),
        ('callback', c_void_p32),
        ('callback2', c_void_p32),
        ('param', c_void_p32),
        ('data', c_void_p32),
        ('ID', c_int),
        ('pnext', POINTER32(ciServerMessageFilter)),
    ]
