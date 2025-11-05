from ctypes import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.types import *


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_race_mapobj_obstacles_pendulum_h_3_9_(Structure):
    _fields_ = [
        ('mobj', c_int),
        ('rotation', c_int),
        ('prevPosition', VecFx32),
        ('renderPos', VecFx32),
        ('shadowMtx', MtxFx43),
        ('offsetY', fx32),
        ('swingRange', u16),
        ('swingVelocity', u16),
        ('angle', u16),
        ('size', VecFx32),
    ]
