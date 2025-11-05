from ctypes import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.types import *


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
