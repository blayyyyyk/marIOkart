from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.animationManager import *
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.list import *
from private.mkds_python_bindings.generated.model import *
from private.mkds_python_bindings.generated.types import *

EXPL_TYPE_BOMBCORE = 0
EXPL_TYPE_BOMBCORE2 = 1
EXPL_TYPE_SPLASH = 2
EXPL_TYPE_BAKUBAKU_WAVE = 3
EXPL_TYPE_MISSION_SPLASH = 4
EXPL_TYPE_COUNT = 5

EXPL_TYPE_BOMBCORE = 0
EXPL_TYPE_BOMBCORE2 = 1
EXPL_TYPE_SPLASH = 2
EXPL_TYPE_BAKUBAKU_WAVE = 3
EXPL_TYPE_MISSION_SPLASH = 4
EXPL_TYPE_COUNT = 5

EXPL_STATE_NORMAL = 0
EXPL_STATE_SHOULD_DIE = 1

EXPL_STATE_NORMAL = 0
EXPL_STATE_SHOULD_DIE = 1


class expl_def_t(Structure):
    _fields_ = [
        ('instanceCount', u16),
        ('hasNsbca', BOOL),
        ('hasNsbma', BOOL),
        ('hasNsbta', BOOL),
        ('initInstFunc', expl_inst_init_func_t),
        ('updateInstFunc', expl_inst_update_func_t),
        ('instanceSize', u32),
        ('modelRes', model_res_t),
    ]

class expl_state_t(Structure):
    _fields_ = [
        ('activeInstanceList', NNSFndList),
        ('freeInstanceLists', (NNSFndList * 5)),
        ('instances', (POINTER32(POINTER32(expl_inst_t)) * 5)),
        ('curPolygonId', u16),
    ]

class expl_inst_t(Structure):
    _fields_ = [
        ('link', NNSFndLink),
        ('model', model_t),
        ('position', VecFx32),
        ('initFunc', expl_inst_init_func_t),
        ('updateFunc', expl_inst_update_func_t),
        ('state', ExplState),
        ('type', ExplType),
        ('nsbcaAnim', POINTER32(anim_manager_t)),
        ('nsbmaAnim', POINTER32(anim_manager_t)),
        ('nsbtaAnim', POINTER32(anim_manager_t)),
        ('scale', VecFx32),
        ('frameCounter', u32),
        ('lifeTime', u32),
        ('polygonId', u16),
        ('visible', BOOL),
    ]
expl_inst_update_func_t = c_void_p32
ExplType = c_int
ExplState = c_int
expl_inst_init_func_t = c_void_p32
