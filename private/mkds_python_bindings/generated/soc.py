from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.ssl import *
from private.mkds_python_bindings.generated.types import *

SOCSslConnection = CPSSslConnection
SOCCaInfo = CPSCaInfo

class SOCSockAddr(Structure):
    _fields_ = [
        ('len', u8),
        ('family', u8),
        ('data', (u8 * 6)),
    ]

class SOCSockAddrIn(Structure):
    _fields_ = [
        ('len', u8),
        ('family', u8),
        ('port', u16),
        ('addr', SOCInAddr),
    ]

class SOCPollFD(Structure):
    _fields_ = [
        ('fd', c_int),
        ('events', c_short),
        ('revents', c_short),
    ]

class SOCHostEnt(Structure):
    _fields_ = [
        ('name', POINTER32(c_char)),
        ('aliases', POINTER32(POINTER32(c_char))),
        ('addrType', s16),
        ('length', s16),
        ('addrList', POINTER32(POINTER32(u8))),
    ]

class SOCIPSocket(Structure):
    _fields_ = [
        ('len', u8),
        ('family', u8),
        ('port', u16),
        ('addr', (u8 * 4)),
    ]

class SOCConfig(Structure):
    _fields_ = [
        ('vendor', u16),
        ('version', u16),
        ('alloc', c_void_p32),
        ('free', c_void_p32),
        ('flag', u32),
        ('addr', SOCInAddr),
        ('netmask', SOCInAddr),
        ('router', SOCInAddr),
        ('dns1', SOCInAddr),
        ('dns2', SOCInAddr),
        ('timeWaitBuffer', s32),
        ('reassemblyBuffer', s32),
        ('mtu', s32),
        ('rwin', s32),
        ('r2', SOCTime),
        ('peerid', POINTER32(c_char)),
        ('passwd', POINTER32(c_char)),
        ('serviceName', POINTER32(c_char)),
        ('hostName', POINTER32(c_char)),
        ('rdhcp', s32),
        ('udpSendBuff', s32),
        ('udpRecvBuff', s32),
    ]

class SOCAddrInfo(Structure):
    _fields_ = [
        ('flags', c_int),
        ('family', c_int),
        ('sockType', c_int),
        ('protocol', c_int),
        ('addrLen', c_int),
        ('canonName', POINTER32(c_char)),
        ('addr', c_void_p32),
        ('next', POINTER32(SOCAddrInfo)),
    ]

class SOCInterface(Structure):
    _fields_ = [
        ('_dummy', u32),
    ]

class SOCDHCPInfo(Structure):
    _fields_ = [
        ('dummy', u32),
    ]
SOCSslAuthHandler = c_void_p32
SOCTime = c_long

class SOCInAddr(Structure):
    _fields_ = [
        ('addr', u32),
    ]
