from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.mapobjInstance import *
from private.mkds_python_bindings.generated.types import *


class rainstar_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('nsbtaFrame', u16),
    ]
