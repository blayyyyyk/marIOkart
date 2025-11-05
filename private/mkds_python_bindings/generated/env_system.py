from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.types import *


class ENVResourceIter(Structure):
    _fields_ = [
        ('link', POINTER32(ENVResourceSetLink)),
        ('ptr', POINTER32(ENVResource)),
        ('count', c_int),
        ('supData', c_void_p32),
    ]

class ENVResourceSetLink(Structure):
    _fields_ = [
        ('next', POINTER32(ENVResourceSetLink)),
        ('array', POINTER32(ENVResource)),
    ]

class ENVResource(Structure):
    _fields_ = [
        ('name', POINTER32(c_char)),
        ('len', u16),
        ('type', ENVType),
        ('ptr', c_void_p32),
    ]
ENVType = c_short
