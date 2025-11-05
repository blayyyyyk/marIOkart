from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.mapobjInstance import *
from private.mkds_python_bindings.generated.types import *


class bsfx_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('sfxAMaxVolume', c_int),
        ('volume', c_int),
        ('fieldA8', c_int),
        ('position', VecFx32),
        ('fieldB8', c_int),
        ('sfxIdA', u16),
        ('sfxIdB', u16),
        ('stateUpdateCounter', c_int),
        ('state', c_int),
        ('sfxACounter', c_int),
    ]
