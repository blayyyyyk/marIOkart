from ctypes import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.gxcommon import *
from private.mkds_python_bindings.testing.spaList import *
from private.mkds_python_bindings.testing.types import *


class spa_particle_t(Structure):
    _fields_ = [
        ('listLink', spa_list_link_t),
        ('position', VecFx32),
        ('velocity', VecFx32),
        ('rotation', u16),
        ('rotationVelocity', s16),
        ('maxAge', u16),
        ('age', u16),
        ('progressSpeedLoop', u16),
        ('progressSpeedNoLoop', u16),
        ('textureId', u8),
        ('progressOffset', u8),
        ('baseAlpha', u16),
        ('alpha', u16),
        ('polygonId', u16),
        ('baseScale', fx32),
        ('scale', fx16),
        ('color', GXRgb),
        ('basePosition', VecFx32),
    ]
