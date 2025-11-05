from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.mapobjInstance import *
from private.mkds_python_bindings.generated.types import *


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
