from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.dynamicCollision import *


class mrbarrier_t(Structure):
    _fields_ = [
        ('dcolMObj', dcol_inst_t),
    ]
