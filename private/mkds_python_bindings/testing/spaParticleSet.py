from ctypes import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.spaEmitter import *
from private.mkds_python_bindings.testing.spaList import *
from private.mkds_python_bindings.testing.spaRes import *
from private.mkds_python_bindings.testing.types import *

spa_alloc_func_t = u32
spa_alloc_func_t = u32
spa_alloc_func_t = u32
spa_alloc_func_t = u32
spa_alloc_func_t = u32
spa_alloc_func_t = u32
spa_alloc_func_t = u32
spa_alloc_func_t = u32
spa_alloc_func_t = u32
spa_alloc_func_t = u32
spa_alloc_func_t = u32
spa_alloc_func_t = u32
spa_alloc_func_t = u32
spa_alloc_func_t = u32
spa_emitter_create_callback_func_t = u32
spa_emitter_create_callback_func_t = u32
spa_emitter_create_callback_func_t = u32
spa_emitter_create_callback_func_t = u32
spa_emitter_create_callback_func_t = u32
spa_emitter_create_callback_func_t = u32
spa_emitter_create_callback_func_t = u32
spa_emitter_create_callback_func_t = u32
spa_emitter_create_callback_func_t = u32
spa_emitter_create_callback_func_t = u32
spa_emitter_create_callback_func_t = u32
spa_emitter_create_callback_func_t = u32
spa_emitter_create_callback_func_t = u32
spa_emitter_create_callback_func_t = u32


class spa_transform_func_arg_pair_t(Structure):
    _fields_ = [
        ('func', u32),
        ('resData', u32),
    ]

class spa_emitter_data_t(Structure):
    _fields_ = [
        ('resource', u32), #POINTER(spa_res_emitter_t)),
        ('scaleAnim', u32), #POINTER(spa_res_emitter_scaleanim_t)),
        ('colorAnim', u32), #POINTER(spa_res_emitter_coloranim_t)),
        ('alphaAnim', u32), #POINTER(spa_res_emitter_alphaanim_t)),
        ('texAnim', u32), #POINTER(spa_res_emitter_texanim_t)),
        ('childData', u32), #POINTER(spa_res_emitter_child_t)),
        ('fieldFuncs', u32), #POINTER(spa_transform_func_arg_pair_t)),
        ('fieldFuncCount', u16),
    ]

class spa_texture_data_t(Structure):
    _fields_ = [
        ('resource', u32), #POINTER(spa_res_texture_t)),
        ('texVramAddr', u32),
        ('plttVramAddr', u32),
        ('texParams', spa_res_texture_params_t),
        ('width', u16),
        ('height', u16),
    ]

class spa_particleset_t(Structure):
    _fields_ = [
        ('allocFunc', spa_alloc_func_t),
        ('activeEmitterList', spa_list_t),
        ('freeEmitterList', spa_list_t),
        ('freeParticleList', spa_list_t),
        ('emitterData', u32), #POINTER(spa_emitter_data_t)),
        ('textureData', u32), #POINTER(spa_texture_data_t)),
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
        ('curEmitter', u32), #POINTER(spa_emitter_t)),
        ('cameraMtx', u32), #POINTER(MtxFx43)),
        ('frameIndex', u16),
    ]
