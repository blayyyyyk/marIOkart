from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.rotDieMObj import *
from private.mkds_python_bindings.generated.types import *


class obpakkunsf_t(Structure):
    _fields_ = [
        ('rotDieMObj', rotdiemobj_t),
        ('counter', u16),
    ]
