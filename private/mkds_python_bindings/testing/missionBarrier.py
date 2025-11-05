from ctypes import *
from private.mkds_python_bindings.testing.dynamicCollision import *


class mrbarrier_t(Structure):
    _fields_ = [
        ('dcolMObj', dcol_inst_t),
    ]
