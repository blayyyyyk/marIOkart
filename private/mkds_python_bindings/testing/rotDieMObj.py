from ctypes import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.mapobjInstance import *
from private.mkds_python_bindings.testing.types import *


class rotdiemobj_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('dieMinY', fx32),
        ('dieYAccel', fx32),
        ('dieRotZDir', BOOL),
        ('dieRotZ', u16),
        ('dieRotZSpeed', u16),
        ('dieInitialYVelo', fx32),
        ('fieldB4', fx32),
    ]
