from ctypes import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.types import *


class objshadow_t(Structure):
    _fields_ = [
        ('mtx', MtxFx43),
        ('alpha', u16),
    ]
