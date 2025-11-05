from ctypes import *
from private.mkds_python_bindings.testing.nnsfnd import *
from typing import TypeVar, ParamSpec

P = ParamSpec('P')
T = TypeVar('T')

class list_link_t(Structure): # 8
    _fields_ = [
        ('link', NNSFndLink), # 4
        ('list', c_int), # 4
    ]
