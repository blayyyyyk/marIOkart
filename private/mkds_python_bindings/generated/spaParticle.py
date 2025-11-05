from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.spaList import *


class spa_particle_t(Structure):
    _fields_ = [
        ('listLink', spa_list_link_t),
        ('position', c_int),
        ('velocity', c_int),
        ('rotation', c_int),
        ('rotationVelocity', c_int),
        ('maxAge', c_int),
        ('age', c_int),
        ('progressSpeedLoop', c_int),
        ('progressSpeedNoLoop', c_int),
        ('textureId', c_int),
        ('progressOffset', c_int),
        ('baseAlpha', c_int),
        ('alpha', c_int),
        ('polygonId', c_int),
        ('baseScale', c_int),
        ('scale', c_int),
        ('color', c_int),
        ('basePosition', c_int),
    ]
