from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.gxcommon import *
from private.mkds_python_bindings.generated.types import *


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_jnlib_spa_spaRes_h_31_9(Structure):
    _fields_ = [
        ('emitterShape', c_int),
        ('particleType', c_int),
        ('axisDirType', c_int),
        ('hasScaleAnim', c_int),
        ('hasColorAnim', c_int),
        ('hasAlphaAnim', c_int),
        ('hasTexAnim', c_int),
        ('hasRandomParticleDeltaRotation', c_int),
        ('hasRandomParticleRotation', c_int),
        ('emitterIsOneTime', c_int),
        ('particlesFollowEmitter', c_int),
        ('hasChildParticles', c_int),
        ('rotMtxMode', c_int),
        ('quadDirection', c_int),
        ('randomizeParticleProgressOffset', c_int),
        ('renderChildParticlesFirst', c_int),
        ('dontRenderMainParticles', c_int),
        ('relativePosAsRotOrigin', c_int),
        ('hasFieldGravity', c_int),
        ('hasFieldRandom', c_int),
        ('hasFieldMagnetic', c_int),
        ('hasFieldSpin', c_int),
        ('hasFieldCollision', c_int),
        ('hasFieldConvergence', c_int),
        ('useConstPolygonIdForMainParticles', c_int),
        ('useConstPolygonIdForChildParticles', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_jnlib_spa_spaRes_h_61_9(Structure):
    _fields_ = [
        ('flags', spa_res_emitter_flags_t),
        ('position', c_int),
        ('emissionCount', c_int),
        ('emitterRadius', c_int),
        ('emitterLength', c_int),
        ('emitterAxis', c_int),
        ('color', c_int),
        ('particlePosVeloMag', c_int),
        ('particleAxisVeloMag', c_int),
        ('particleBaseScale', c_int),
        ('aspectRatio', c_int),
        ('emissionStartTime', c_int),
        ('minRotVelocity', c_int),
        ('maxRotVelocity', c_int),
        ('particleRotation', c_int),
        ('padding1', c_int),
        ('emissionTime', c_int),
        ('particleLifetime', c_int),
        ('particleScaleRandomness', c_int),
        ('particleLifetimeRandomness', c_int),
        ('particleVeloMagRandomness', c_int),
        ('padding2', c_int),
        ('emissionInterval', c_int),
        ('particleAlpha', c_int),
        ('airResistance', c_int),
        ('textureId', c_int),
        ('loopFrame', c_int),
        ('dirBillboardScale', c_int),
        ('texRepeatShiftS', c_int),
        ('texRepeatShiftT', c_int),
        ('scaleMode', c_int),
        ('centerDirPolygon', c_int),
        ('texFlipS', c_int),
        ('texFlipT', c_int),
        ('u32', c_int),
        ('quadXOffset', c_int),
        ('quadYZOffset', c_int),
    ]

class union__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_jnlib_spa_spaRes_h_104_5(Union):
    _fields_ = [
        ('userDataU32', c_int),
        ('userDataU16', (c_int * 2)),
        ('userDataU8', (c_int * 4)),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_jnlib_spa_spaRes_h_114_9(Structure):
    _fields_ = [
        ('initialScale', c_int),
        ('intermediateScale', c_int),
        ('endingScale', c_int),
        ('inCompletedTiming', c_int),
        ('scaleOutStartTime', c_int),
        ('loop', c_int),
        ('u16', c_int),
        ('padding', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_jnlib_spa_spaRes_h_128_9(Structure):
    _fields_ = [
        ('initialColor', c_int),
        ('endingColor', c_int),
        ('inCompletedTiming', c_int),
        ('peakTiming', c_int),
        ('outStartTiming', c_int),
        ('field7', c_int),
        ('isRandom', c_int),
        ('loop', c_int),
        ('interpolate', c_int),
        ('u16', c_int),
        ('padding', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_jnlib_spa_spaRes_h_145_9(Structure):
    _fields_ = [
        ('initialAlpha', c_int),
        ('peakAlpha', c_int),
        ('endingAlpha', c_int),
        ('u16', c_int),
        ('randomness', c_int),
        ('loop', c_int),
        ('inCompletedTiming', c_int),
        ('outStartTiming', c_int),
        ('padding', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_jnlib_spa_spaRes_h_161_9(Structure):
    _fields_ = [
        ('frames', (c_int * 8)),
        ('frameCount', c_int),
        ('frameDuration', c_int),
        ('isRandom', c_int),
        ('loop', c_int),
        ('u32', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_jnlib_spa_spaRes_h_177_9(Structure):
    _fields_ = [
        ('applyEmitterField', c_int),
        ('useScaleAnim', c_int),
        ('hasAlphaFade', c_int),
        ('rotInheritMode', c_int),
        ('followEmitter', c_int),
        ('useChildColor', c_int),
        ('particleType', c_int),
        ('rotMtxMode', c_int),
        ('quadDirection', c_int),
        ('u16', c_int),
        ('initVelocityRandomness', c_int),
        ('targetScale', c_int),
        ('lifeTime', c_int),
        ('velocityInheritRatio', c_int),
        ('scale', c_int),
        ('color', c_int),
        ('emissionVolume', c_int),
        ('emissionTime', c_int),
        ('emissionInterval', c_int),
        ('textureId', c_int),
        ('texRepeatShiftS', c_int),
        ('texRepeatShiftT', c_int),
        ('texFlipS', c_int),
        ('texFlipT', c_int),
        ('centerDirPolygon', c_int),
        ('u32', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_jnlib_spa_spaRes_h_211_9(Structure):
    _fields_ = [
        ('gravity', c_int),
        ('padding', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_jnlib_spa_spaRes_h_219_9(Structure):
    _fields_ = [
        ('strength', c_int),
        ('interval', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_jnlib_spa_spaRes_h_227_9(Structure):
    _fields_ = [
        ('magnetPos', c_int),
        ('magnetPower', c_int),
        ('padding', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_jnlib_spa_spaRes_h_240_9(Structure):
    _fields_ = [
        ('rotation', c_int),
        ('type', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_jnlib_spa_spaRes_h_251_9(Structure):
    _fields_ = [
        ('collisionPlaneY', c_int),
        ('bounceCoef', c_int),
        ('behavior', c_int),
        ('u16', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_jnlib_spa_spaRes_h_261_9(Structure):
    _fields_ = [
        ('convergencePos', c_int),
        ('convergenceRatio', c_int),
        ('padding', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_jnlib_spa_spaRes_h_270_9(Structure):
    _fields_ = [
        ('format', c_int),
        ('width', c_int),
        ('height', c_int),
        ('repeat', c_int),
        ('flip', c_int),
        ('pltt0Transparent', c_int),
        ('refTexData', c_int),
        ('refTexId', c_int),
        ('u32', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_jnlib_spa_spaRes_h_283_9(Structure):
    _fields_ = [
        ('signature', c_int),
        ('texParams', spa_res_texture_params_t),
        ('texSize', c_int),
        ('plttOffset', c_int),
        ('plttSize', c_int),
        ('plttIdxOffset', c_int),
        ('plttIdxSize', c_int),
        ('blockSize', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_jnlib_spa_spaRes_h_297_9(Structure):
    _fields_ = [
        ('signature', c_int),
        ('version', c_int),
        ('emitterCount', c_int),
        ('textureCount', c_int),
        ('fieldC', c_int),
        ('emitterBlockLength', c_int),
        ('textureBlockLength', c_int),
        ('textureBlockOffset', c_int),
        ('field1C', c_int),
    ]

class union__anonymous_at__Users_blakemoody_dev_mariokart_ml_mkds_c_jnlib_spa_spaRes_h_104_5(Union):
    _fields_ = [
        ('userDataU32', u32),
        ('userDataU16', (u16 * 2)),
        ('userDataU8', (u8 * 4)),
    ]

class spa_res_emitter_field_gravity_t(Structure):
    _fields_ = [
        ('gravity', VecFx16),
        ('padding', u16),
    ]

class spa_res_emitter_field_random_t(Structure):
    _fields_ = [
        ('strength', VecFx16),
        ('interval', u16),
    ]

class spa_res_emitter_field_magnet_t(Structure):
    _fields_ = [
        ('magnetPos', VecFx32),
        ('magnetPower', fx16),
        ('padding', u16),
    ]

class spa_res_emitter_field_spin_t(Structure):
    _fields_ = [
        ('rotation', u16),
        ('type', u16),
    ]

class spa_res_emitter_field_collision_t(Structure):
    _fields_ = [
        ('collisionPlaneY', fx32),
        ('bounceCoef', fx16),
        ('behavior', u16),
        ('_anon', u16),
    ]

class spa_res_emitter_field_convergence_t(Structure):
    _fields_ = [
        ('convergencePos', VecFx32),
        ('convergenceRatio', fx16),
        ('padding', u16),
    ]

class spa_res_header_t(Structure):
    _fields_ = [
        ('signature', u32),
        ('version', u32),
        ('emitterCount', u16),
        ('textureCount', u16),
        ('fieldC', u32),
        ('emitterBlockLength', u32),
        ('textureBlockLength', u32),
        ('textureBlockOffset', u32),
        ('field1C', u32),
    ]

class spa_res_texture_t(Structure):
    _fields_ = [
        ('signature', u32),
        ('texParams', spa_res_texture_params_t),
        ('texSize', u32),
        ('plttOffset', u32),
        ('plttSize', u32),
        ('plttIdxOffset', u32),
        ('plttIdxSize', u32),
        ('blockSize', u32),
    ]

class spa_res_texture_params_t(Structure):
    _fields_ = [
        ('format', u32),
        ('width', u32),
        ('height', u32),
        ('repeat', u32),
        ('flip', u32),
        ('pltt0Transparent', u32),
        ('refTexData', u32),
        ('refTexId', u32),
        ('_anon', u32),
    ]

class spa_res_emitter_scaleanim_t(Structure):
    _fields_ = [
        ('initialScale', fx16),
        ('intermediateScale', fx16),
        ('endingScale', fx16),
        ('inCompletedTiming', u8),
        ('scaleOutStartTime', u8),
        ('loop', u16),
        ('_anon', u16),
        ('padding', u16),
    ]

class spa_res_emitter_texanim_t(Structure):
    _fields_ = [
        ('frames', (u8 * 8)),
        ('frameCount', u32),
        ('frameDuration', u32),
        ('isRandom', u32),
        ('loop', u32),
        ('_anon', u32),
    ]

class spa_res_emitter_coloranim_t(Structure):
    _fields_ = [
        ('initialColor', GXRgb),
        ('endingColor', GXRgb),
        ('inCompletedTiming', u8),
        ('peakTiming', u8),
        ('outStartTiming', u8),
        ('field7', u8),
        ('isRandom', u16),
        ('loop', u16),
        ('interpolate', u16),
        ('_anon', u16),
        ('padding', u16),
    ]

class spa_res_emitter_alphaanim_t(Structure):
    _fields_ = [
        ('initialAlpha', u16),
        ('peakAlpha', u16),
        ('endingAlpha', u16),
        ('_anon', u16),
        ('randomness', u16),
        ('loop', u16),
        ('_anon', u16),
        ('inCompletedTiming', u8),
        ('outStartTiming', u8),
        ('padding', u16),
    ]

class spa_res_emitter_child_t(Structure):
    _fields_ = [
        ('applyEmitterField', u16),
        ('useScaleAnim', u16),
        ('hasAlphaFade', u16),
        ('rotInheritMode', u16),
        ('followEmitter', u16),
        ('useChildColor', u16),
        ('particleType', u16),
        ('rotMtxMode', u16),
        ('quadDirection', u16),
        ('_anon', u16),
        ('initVelocityRandomness', s16),
        ('targetScale', fx16),
        ('lifeTime', u16),
        ('velocityInheritRatio', u8),
        ('scale', u8),
        ('color', GXRgb),
        ('emissionVolume', u8),
        ('emissionTime', u8),
        ('emissionInterval', u8),
        ('textureId', u8),
        ('texRepeatShiftS', u32),
        ('texRepeatShiftT', u32),
        ('texFlipS', u32),
        ('texFlipT', u32),
        ('centerDirPolygon', u32),
        ('_anon', u32),
    ]

class spa_res_emitter_t(Structure):
    _fields_ = [
        ('flags', spa_res_emitter_flags_t),
        ('position', VecFx32),
        ('emissionCount', fx32),
        ('emitterRadius', fx32),
        ('emitterLength', fx32),
        ('emitterAxis', VecFx16),
        ('color', GXRgb),
        ('particlePosVeloMag', fx32),
        ('particleAxisVeloMag', fx32),
        ('particleBaseScale', fx32),
        ('aspectRatio', fx16),
        ('emissionStartTime', u16),
        ('minRotVelocity', s16),
        ('maxRotVelocity', s16),
        ('particleRotation', u16),
        ('padding1', u16),
        ('emissionTime', u16),
        ('particleLifetime', u16),
        ('particleScaleRandomness', u8),
        ('particleLifetimeRandomness', u8),
        ('particleVeloMagRandomness', u8),
        ('padding2', u8),
        ('emissionInterval', u8),
        ('particleAlpha', u8),
        ('airResistance', u8),
        ('textureId', u8),
        ('loopFrame', u32),
        ('dirBillboardScale', u32),
        ('texRepeatShiftS', u32),
        ('texRepeatShiftT', u32),
        ('scaleMode', u32),
        ('centerDirPolygon', u32),
        ('texFlipS', u32),
        ('texFlipT', u32),
        ('_anon', u32),
        ('quadXOffset', fx16),
        ('quadYZOffset', fx16),
    ]

class spa_res_emitter_flags_t(Structure):
    _fields_ = [
        ('emitterShape', u32),
        ('particleType', u32),
        ('axisDirType', u32),
        ('hasScaleAnim', u32),
        ('hasColorAnim', u32),
        ('hasAlphaAnim', u32),
        ('hasTexAnim', u32),
        ('hasRandomParticleDeltaRotation', u32),
        ('hasRandomParticleRotation', u32),
        ('emitterIsOneTime', u32),
        ('particlesFollowEmitter', u32),
        ('hasChildParticles', u32),
        ('rotMtxMode', u32),
        ('quadDirection', u32),
        ('randomizeParticleProgressOffset', u32),
        ('renderChildParticlesFirst', u32),
        ('dontRenderMainParticles', u32),
        ('relativePosAsRotOrigin', u32),
        ('hasFieldGravity', u32),
        ('hasFieldRandom', u32),
        ('hasFieldMagnetic', u32),
        ('hasFieldSpin', u32),
        ('hasFieldCollision', u32),
        ('hasFieldConvergence', u32),
        ('useConstPolygonIdForMainParticles', u32),
        ('useConstPolygonIdForChildParticles', u32),
    ]
