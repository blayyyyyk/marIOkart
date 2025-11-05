from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.spaList import *
from private.mkds_python_bindings.generated.spaParticleSet import *


class union__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_jnlib_spa_spaEmitter_h_23_5(Union):
    _fields_ = [
        ('flags', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_jnlib_spa_spaEmitter_h_27_9(Structure):
    _fields_ = [
        ('shouldDie', c_int),
        ('disableEmission', c_int),
        ('disableUpdates', c_int),
        ('disableRendering', c_int),
        ('startedEmitting', c_int),
        ('u32', c_int),
    ]

class spa_emitter_t(Structure):
    _fields_ = [
        ('listLink', spa_list_link_t),
        ('mainParticleList', spa_list_t),
        ('childParticleList', spa_list_t),
        ('emitterData', POINTER32(spa_emitter_data_t)),
        ('position', c_int),
        ('velocity', c_int),
        ('particleVelocity', c_int),
        ('age', c_int),
        ('lastEmissionFraction', c_int),
        ('axis', c_int),
        ('particleRotation', c_int),
        ('emissionCount', c_int),
        ('radius', c_int),
        ('length', c_int),
        ('particlePosVeloMag', c_int),
        ('particleAxisVeloMag', c_int),
        ('particleScale', c_int),
        ('particleLifetime', c_int),
        ('color', c_int),
        ('collisionPlaneYOverride', c_int),
        ('texS', c_int),
        ('texT', c_int),
        ('childTexS', c_int),
        ('childTexT', c_int),
        ('emissionInterval', c_int),
        ('particleAlpha', c_int),
        ('updateMoment', c_int),
        ('field80Unk2', c_int),
        ('axisRight', c_int),
        ('axisUp', c_int),
        ('callbackFunc', spa_emitter_callback_func_t),
        ('userData', c_int),
    ]
spa_emitter_callback_func_t = c_void_p32
