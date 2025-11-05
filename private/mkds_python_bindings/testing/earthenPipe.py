from ctypes import *
from private.mkds_python_bindings.testing.rotDieMObj import *


class epipe_t(Structure):
    _fields_ = [
        ('rotDieMObj', rotdiemobj_t),
    ]
