from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.rotDieMObj import *


class epipe_t(Structure):
    _fields_ = [
        ('rotDieMObj', rotdiemobj_t),
    ]
