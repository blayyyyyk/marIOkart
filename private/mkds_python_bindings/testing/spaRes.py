from ctypes import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.gxcommon import *
from private.mkds_python_bindings.testing.types import *


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
        ('_', u32),
        ('quadXOffset', fx16),
        ('quadYZOffset', fx16),
    ]

class union__anonymous_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_jnlib_spa_spaRes_h_104_5_(Union):
    _fields_ = [
        ('userDataU32', u32),
        ('userDataU16', (u16 * 2)),
        ('userDataU8', (u8 * 4)),
    ]

class spa_res_emitter_scaleanim_t(Structure):
    _fields_ = [
        ('initialScale', fx16),
        ('intermediateScale', fx16),
        ('endingScale', fx16),
        ('inCompletedTiming', u8),
        ('scaleOutStartTime', u8),
        ('loop', u16),
        ('_', u16),
        ('padding', u16),
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
        ('_', u16),
        ('padding', u16),
    ]

class spa_res_emitter_alphaanim_t(Structure):
    _fields_ = [
        ('initialAlpha', u16),
        ('peakAlpha', u16),
        ('endingAlpha', u16),
        ('_', u16),
        ('randomness', u16),
        ('loop', u16),
        ('_', u16),
        ('inCompletedTiming', u8),
        ('outStartTiming', u8),
        ('padding', u16),
    ]

class spa_res_emitter_texanim_t(Structure):
    _fields_ = [
        ('frames', (u8 * 8)),
        ('frameCount', u32),
        ('frameDuration', u32),
        ('isRandom', u32),
        ('loop', u32),
        ('_', u32),
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
        ('_', u16),
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
        ('_', u32),
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
        ('_', u16),
    ]

class spa_res_emitter_field_convergence_t(Structure):
    _fields_ = [
        ('convergencePos', VecFx32),
        ('convergenceRatio', fx16),
        ('padding', u16),
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
        ('_', u32),
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
