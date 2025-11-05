from ctypes import *
from private.mkds_python_bindings.testing.mapobjInstance import *
from private.mkds_python_bindings.testing.types import *


class fwdst_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('index', u16),
        ('ccDependentSetting', u16),
    ]
