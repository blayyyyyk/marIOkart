from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.mapobjInstance import *
from private.mkds_python_bindings.generated.types import *


class fireball2_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('nrArms', u16),
        ('fireballsPerArm', u16),
        ('armAngleDistance', u16),
        ('fireballDistance', fx32),
        ('radius', fx32),
        ('rotSpeed', u16),
        ('rotation', u16),
        ('centerFireball', fireball2_fireball_t),
        ('fireballs', ((fireball2_fireball_t * 20) * 20)),
        ('driverHitTimeouts', (c_int * 8)),
        ('playerDistanceFromRing', fx32),
    ]

class fireball2_fireball_t(Structure):
    _fields_ = [
        ('position', VecFx32),
        ('armRotZ', u16),
        ('ballRotZ', u16),
    ]
