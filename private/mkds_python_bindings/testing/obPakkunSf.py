from ctypes import *
from private.mkds_python_bindings.testing.rotDieMObj import *
from private.mkds_python_bindings.testing.types import *


class obpakkunsf_t(Structure):
    _fields_ = [
        ('rotDieMObj', rotdiemobj_t),
        ('counter', u16),
    ]
