from ctypes import *
from private.mkds_python_bindings.testing.types import *


class spa_list_link_t(Structure):
    _fields_ = [
        ('next', u32), #POINTER(spa_list_link_t)),
        ('prev', u32), #POINTER(spa_list_link_t)),
    ]

class spa_list_t(Structure):
    _fields_ = [
        ('head', u32), #POINTER(spa_list_link_t)),
        ('count', u32),
        ('tail', u32), #POINTER(spa_list_link_t)),
    ]
