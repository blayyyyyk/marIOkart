from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.spaEmitter import *
from private.mkds_python_bindings.generated.spaList import *
from private.mkds_python_bindings.generated.spaRes import *
from private.mkds_python_bindings.generated.types import *


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_jnlib_spa_spaParticleSet_h_18_9(Structure):
    _fields_ = [
        ('resource', POINTER32(spa_res_emitter_t)),
        ('scaleAnim', POINTER32(spa_res_emitter_scaleanim_t)),
        ('colorAnim', POINTER32(spa_res_emitter_coloranim_t)),
        ('alphaAnim', POINTER32(spa_res_emitter_alphaanim_t)),
        ('texAnim', POINTER32(spa_res_emitter_texanim_t)),
        ('childData', POINTER32(spa_res_emitter_child_t)),
        ('fieldFuncs', POINTER32(spa_transform_func_arg_pair_t)),
        ('fieldFuncCount', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_jnlib_spa_spaParticleSet_h_30_9(Structure):
    _fields_ = [
        ('resource', POINTER32(spa_res_texture_t)),
        ('texVramAddr', c_int),
        ('plttVramAddr', c_int),
        ('texParams', spa_res_texture_params_t),
        ('width', c_int),
        ('height', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_jnlib_spa_spaParticleSet_h_40_9(Structure):
    _fields_ = [
        ('allocFunc', spa_alloc_func_t),
        ('activeEmitterList', spa_list_t),
        ('freeEmitterList', spa_list_t),
        ('freeParticleList', spa_list_t),
        ('emitterData', POINTER32(spa_emitter_data_t)),
        ('textureData', POINTER32(spa_texture_data_t)),
        ('resEmitterCount', c_int),
        ('resTexCount', c_int),
        ('maxEmitterCount', c_int),
        ('maxParticleCount', c_int),
        ('firstPolygonId', c_int),
        ('lastPolygonId', c_int),
        ('curPolygonId', c_int),
        ('constPolygonId', c_int),
        ('reverseRenderOrder', c_int),
        ('unknown', c_int),
        ('polygonAttr', c_int),
        ('curEmitter', POINTER32(spa_emitter_t)),
        ('cameraMtx', POINTER32(c_int)),
        ('frameIndex', c_int),
    ]

class spa_particleset_t(Structure):
    _fields_ = [
        ('allocFunc', spa_alloc_func_t),
        ('activeEmitterList', spa_list_t),
        ('freeEmitterList', spa_list_t),
        ('freeParticleList', spa_list_t),
        ('emitterData', POINTER32(spa_emitter_data_t)),
        ('textureData', POINTER32(spa_texture_data_t)),
        ('resEmitterCount', u16),
        ('resTexCount', u16),
        ('maxEmitterCount', u16),
        ('maxParticleCount', u16),
        ('firstPolygonId', u32),
        ('lastPolygonId', u32),
        ('curPolygonId', u32),
        ('constPolygonId', u32),
        ('reverseRenderOrder', u32),
        ('unknown', u32),
        ('polygonAttr', u32),
        ('curEmitter', POINTER32(spa_emitter_t)),
        ('cameraMtx', POINTER32(MtxFx43)),
        ('frameIndex', u16),
    ]
spa_emitter_create_callback_func_t = c_void_p32

class spa_texture_data_t(Structure):
    _fields_ = [
        ('resource', POINTER32(spa_res_texture_t)),
        ('texVramAddr', u32),
        ('plttVramAddr', u32),
        ('texParams', spa_res_texture_params_t),
        ('width', u16),
        ('height', u16),
    ]
spa_alloc_func_t = c_void_p32

class spa_emitter_data_t(Structure):
    _fields_ = [
        ('resource', POINTER32(spa_res_emitter_t)),
        ('scaleAnim', POINTER32(spa_res_emitter_scaleanim_t)),
        ('colorAnim', POINTER32(spa_res_emitter_coloranim_t)),
        ('alphaAnim', POINTER32(spa_res_emitter_alphaanim_t)),
        ('texAnim', POINTER32(spa_res_emitter_texanim_t)),
        ('childData', POINTER32(spa_res_emitter_child_t)),
        ('fieldFuncs', POINTER32(spa_transform_func_arg_pair_t)),
        ('fieldFuncCount', u16),
    ]

class spa_transform_func_arg_pair_t(Structure):
    _fields_ = [
        ('func', c_void_p32),
        ('resData', c_void_p32),
    ]
