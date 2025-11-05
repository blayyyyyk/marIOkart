from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.types import *


class NNSFndLink(Structure):
    _fields_ = [
        ('prevObject', c_void_p32),
        ('nextObject', c_void_p32),
    ]

class NNSFndList(Structure):
    _fields_ = [
        ('headObject', c_void_p32),
        ('tailObject', c_void_p32),
        ('numObjects', u16),
        ('offset', u16),
    ]

class list_link_t(Structure):
    _fields_ = [
        ('link', NNSFndLink),
        ('list', POINTER32(NNSFndList)),
    ]
