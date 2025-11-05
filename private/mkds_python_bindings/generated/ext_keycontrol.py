from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.types import *


class EXTKeys(Structure):
    _fields_ = [
        ('key', u16),
        ('count', u16),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_ext_ext_keycontrol_h_33_9(Structure):
    _fields_ = [
        ('key', c_int),
        ('count', c_int),
    ]
