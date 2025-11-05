from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.spaEmitter import *


class dptc_t(Structure):
    _fields_ = [
        ('whiteDustCloudEmitters', (POINTER32(spa_emitter_t) * 2)),
        ('blueSparkEmitters', (POINTER32(spa_emitter_t) * 2)),
        ('contRedSparkEmitters', (POINTER32(spa_emitter_t) * 2)),
        ('contRedSparkSmallEmitters', (POINTER32(spa_emitter_t) * 2)),
        ('redSparkEmitters', (POINTER32(spa_emitter_t) * 2)),
        ('redSparksSmallEmitters', (POINTER32(spa_emitter_t) * 2)),
        ('contRedSparksCounter', c_int),
        ('field34', c_int),
        ('contRedSparksActive', c_int),
        ('whiteDustCloudsActive', c_int),
        ('redSparksActive', c_int),
        ('blueSparksActive', c_int),
        ('redSparksCounter', c_int),
        ('blueSparkCounter', c_int),
        ('whiteDustCloudParticleId', c_int),
        ('contRedSparkParticleId', c_int),
        ('contRedSparkSmallParticleId', c_int),
        ('redSparkParticleId', c_int),
        ('redSparksSmallParticleId', c_int),
        ('driverId', c_int),
        ('field62', c_int),
        ('field64', c_int),
        ('startContinuousRedSparksFunc', dptc_func_t),
        ('suspendContinuousRedSparksFunc', dptc_func_t),
        ('resumeContinuousRedSparksFunc', dptc_func_t),
        ('killContRedSparksFunc', dptc_func_t),
        ('field78', dptc_func_t),
        ('hideAllFunc', dptc_func_t),
        ('showAllFunc', dptc_func_t),
        ('updateContRedSparksFunc', c_void_p32),
    ]
dptc_func_t = c_void_p32
