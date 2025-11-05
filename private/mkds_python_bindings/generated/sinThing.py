from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.types import *


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_sinThing_h_3_9(Structure):
    _fields_ = [
        ('phase', c_int),
        ('value', c_int),
        ('velocity', c_int),
        ('baseOffset', c_int),
        ('amplitude', c_int),
        ('amplitudeVelocity', c_int),
        ('phaseVelocity', c_int),
        ('phaseAcceleration', c_int),
    ]

class sinthing_t(Structure):
    _fields_ = [
        ('phase', u16),
        ('value', fx32),
        ('velocity', fx32),
        ('baseOffset', fx32),
        ('amplitude', fx32),
        ('amplitudeVelocity', fx32),
        ('phaseVelocity', c_int),
        ('phaseAcceleration', c_int),
    ]
