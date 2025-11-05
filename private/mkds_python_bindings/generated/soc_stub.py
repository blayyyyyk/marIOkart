from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.soc import *
from private.mkds_python_bindings.generated.types import *


class ICMPInfo(Structure):
    _fields_ = [
        ('___temporary', c_int),
    ]

class IPHeader(Structure):
    _fields_ = [
        ('verlen', u8),
        ('tos', u8),
        ('len', u16),
        ('id', u16),
        ('frag', u16),
        ('ttl', u8),
        ('proto', u8),
        ('sum', u16),
        ('src', (u8 * 4)),
        ('dst', (u8 * 4)),
    ]

class SOCIpMreq(Structure):
    _fields_ = [
        ('multiaddr', SOCInAddr),
        ('interface', SOCInAddr),
    ]
ICMPCallback = c_void_p32
