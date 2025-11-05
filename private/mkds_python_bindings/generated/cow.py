from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.mapobjInstance import *


class cow_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
    ]
