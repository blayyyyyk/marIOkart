from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.mapobjInstance import *
from private.mkds_python_bindings.generated.quaternion import *
from private.mkds_python_bindings.generated.types import *


class pendulum_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('rotation', quaternion_t),
        ('prevPosition', VecFx32),
        ('renderPos', VecFx32),
        ('shadowMtx', MtxFx43),
        ('offsetY', fx32),
        ('swingRange', u16),
        ('swingVelocity', u16),
        ('angle', u16),
        ('size', VecFx32),
    ]
