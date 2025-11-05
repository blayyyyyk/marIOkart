from ctypes import *
from private.mkds_python_bindings.testing.mapobjInstance import *
from private.mkds_python_bindings.testing.types import *


class rainstar_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('nsbtaFrame', u16),
    ]
