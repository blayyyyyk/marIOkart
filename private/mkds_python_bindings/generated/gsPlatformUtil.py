from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32

class B64StreamData(Structure):
    _fields_ = [
        ('input', POINTER32(c_char)),
        ('len', c_int),
        ('encodingType', c_int),
    ]
GetUniqueIDFunction = c_void_p32
GSIResolveHostnameHandle = c_void_p32
