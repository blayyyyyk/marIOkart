from ctypes import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.spaEmitter import *
from private.mkds_python_bindings.testing.types import *

ptcm_get_particleset_by_id_func_t = u32
ptcm_special_particle_update_func_t = u32


class ptcm_camflashes_t(Structure):
    _fields_ = [
        ('directions', (VecFx32 * 6)),
        ('emitters', (POINTER(spa_emitter_t) * 6)),
        ('waitCounter', s16),
    ]
