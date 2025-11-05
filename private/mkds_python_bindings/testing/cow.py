from ctypes import *
from private.mkds_python_bindings.testing.mapobjInstance import *


class cow_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
    ]
