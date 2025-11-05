from ctypes import *
from private.mkds_python_bindings.testing.types import *

ENVType = c_short


class ENVResource(Structure):
    _fields_ = [
        ('name', u32), #POINTER(c_char)),
        ('len', u16),
        ('type', ENVType),
        ('ptr', u32),
    ]

class ENVResourceSetLink(Structure):
    _fields_ = [
        ('next', u32), #POINTER(ENVResourceSetLink)),
        ('array', u32), #POINTER(ENVResource)),
    ]

class ENVResourceIter(Structure):
    _fields_ = [
        ('link', u32), #POINTER(ENVResourceSetLink)),
        ('ptr', u32), #POINTER(ENVResource)),
        ('count', c_int),
        ('supData', u32),
    ]
