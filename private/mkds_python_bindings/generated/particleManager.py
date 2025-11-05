from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.spaEmitter import *


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_particles_particleManager_h_9_9(Structure):
    _fields_ = [
        ('directions', (c_int * 6)),
        ('emitters', (POINTER32(spa_emitter_t) * 6)),
        ('waitCounter', c_int),
    ]
ptcm_special_particle_update_func_t = c_void_p32
ptcm_get_particleset_by_id_func_t = c_void_p32
