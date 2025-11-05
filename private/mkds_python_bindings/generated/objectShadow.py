from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.types import *


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_objectShadow_h_3_9(Structure):
    _fields_ = [
        ('mtx', c_int),
        ('alpha', c_int),
    ]

class objshadow_t(Structure):
    _fields_ = [
        ('mtx', MtxFx43),
        ('alpha', u16),
    ]
