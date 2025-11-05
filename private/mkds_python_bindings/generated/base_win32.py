from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
NNS_MCS_DEVICE_TYPE_UNKNOWN = 0
NNS_MCS_DEVICE_TYPE_NITRO_DEBUGGER = 1
NNS_MCS_DEVICE_TYPE_NITRO_UIC = 2
NNS_MCS_DEVICE_TYPE_ENSATA = 3


class NNSMcsStreamInfo(Structure):
    _fields_ = [
        ('structBytes', c_int),
        ('deviceType', c_int),
    ]
WINAPI = c_void_p32
HANDLE = c_void_p32
