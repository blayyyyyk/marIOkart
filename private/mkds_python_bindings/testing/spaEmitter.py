from ctypes import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.gxcommon import *
from private.mkds_python_bindings.testing.spaList import *
from private.mkds_python_bindings.testing.spaParticleSet import *
from private.mkds_python_bindings.testing.types import *

spa_emitter_callback_func_t = u32
spa_emitter_callback_func_t = u32
spa_emitter_callback_func_t = u32
spa_emitter_callback_func_t = u32
spa_emitter_callback_func_t = u32
spa_emitter_callback_func_t = u32
spa_emitter_callback_func_t = u32
spa_emitter_callback_func_t = u32
spa_emitter_callback_func_t = u32
spa_emitter_callback_func_t = u32
spa_emitter_callback_func_t = u32
spa_emitter_callback_func_t = u32
spa_emitter_callback_func_t = u32


class spa_emitter_t(Structure):
    _fields_ = [
        ('listLink', spa_list_link_t),
        ('mainParticleList', spa_list_t),
        ('childParticleList', spa_list_t),
        ('emitterData', u32), #POINTER(spa_emitter_data_t)),
        ('position', VecFx32),
        ('velocity', VecFx32),
        ('particleVelocity', VecFx32),
        ('age', u16),
        ('lastEmissionFraction', fx16),
        ('axis', VecFx16),
        ('particleRotation', u16),
        ('emissionCount', fx32),
        ('radius', fx32),
        ('length', fx32),
        ('particlePosVeloMag', fx32),
        ('particleAxisVeloMag', fx32),
        ('particleScale', fx32),
        ('particleLifetime', u16),
        ('color', GXRgb),
        ('collisionPlaneYOverride', fx32),
        ('texS', fx16),
        ('texT', fx16),
        ('childTexS', fx16),
        ('childTexT', fx16),
        ('emissionInterval', u32),
        ('particleAlpha', u32),
        ('updateMoment', u32),
        ('field80Unk2', u32),
        ('axisRight', VecFx16),
        ('axisUp', VecFx16),
        ('callbackFunc', spa_emitter_callback_func_t),
        ('userData', u32),
    ]

class union__anonymous_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_jnlib_spa_spaEmitter_h_23_5_(Union):
    _fields_ = [
        ('flags', u32),
    ]

class struct__anonymous_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_jnlib_spa_spaEmitter_h_27_9_(Structure):
    _fields_ = [
        ('shouldDie', u32),
        ('disableEmission', u32),
        ('disableUpdates', u32),
        ('disableRendering', u32),
        ('startedEmitting', u32),
        ('_', u32),
    ]
