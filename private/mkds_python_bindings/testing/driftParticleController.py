from ctypes import *
from private.mkds_python_bindings.testing.spaEmitter import *
from private.mkds_python_bindings.testing.types import *

dptc_func_t = u32


class dptc_t(Structure):
    _fields_ = [
        ('whiteDustCloudEmitters', (POINTER(spa_emitter_t) * 2)),
        ('blueSparkEmitters', (POINTER(spa_emitter_t) * 2)),
        ('contRedSparkEmitters', (POINTER(spa_emitter_t) * 2)),
        ('contRedSparkSmallEmitters', (POINTER(spa_emitter_t) * 2)),
        ('redSparkEmitters', (POINTER(spa_emitter_t) * 2)),
        ('redSparksSmallEmitters', (POINTER(spa_emitter_t) * 2)),
        ('contRedSparksCounter', s16),
        ('field34', c_int),
        ('contRedSparksActive', BOOL),
        ('whiteDustCloudsActive', BOOL),
        ('redSparksActive', BOOL),
        ('blueSparksActive', BOOL),
        ('redSparksCounter', s16),
        ('blueSparkCounter', s16),
        ('whiteDustCloudParticleId', c_int),
        ('contRedSparkParticleId', c_int),
        ('contRedSparkSmallParticleId', c_int),
        ('redSparkParticleId', c_int),
        ('redSparksSmallParticleId', c_int),
        ('driverId', u16),
        ('field62', u16),
        ('field64', c_int),
        ('startContinuousRedSparksFunc', dptc_func_t),
        ('suspendContinuousRedSparksFunc', dptc_func_t),
        ('resumeContinuousRedSparksFunc', dptc_func_t),
        ('killContRedSparksFunc', dptc_func_t),
        ('field78', dptc_func_t),
        ('hideAllFunc', dptc_func_t),
        ('showAllFunc', dptc_func_t),
        ('updateContRedSparksFunc', u32),
    ]
