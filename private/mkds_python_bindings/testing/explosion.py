from ctypes import *
from private.mkds_python_bindings.testing.animationManager import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.model import *
from private.mkds_python_bindings.testing.nnsfnd import *
from private.mkds_python_bindings.testing.types import *

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

ExplState = c_int
ExplType = c_int
expl_inst_init_func_t = u32
expl_inst_update_func_t = u32


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_race_effects_explosion_h_27_9_(Structure):
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

class expl_inst_t(Structure):
    _fields_ = [
        ('link', NNSFndLink),
        ('model', model_t),
        ('position', VecFx32),
        ('initFunc', expl_inst_init_func_t),
        ('updateFunc', expl_inst_update_func_t),
        ('state', ExplState),
        ('type', ExplType),
        ('nsbcaAnim', u32), #POINTER(anim_manager_t)),
        ('nsbmaAnim', u32), #POINTER(anim_manager_t)),
        ('nsbtaAnim', u32), #POINTER(anim_manager_t)),
        ('scale', VecFx32),
        ('frameCounter', u32),
        ('lifeTime', u32),
        ('polygonId', u16),
        ('visible', BOOL),
    ]

class expl_state_t(Structure):
    _fields_ = [
        ('activeInstanceList', NNSFndList),
        ('freeInstanceLists', (NNSFndList * 5)),
        ('instances', (POINTER(POINTER(expl_inst_t)) * 5)),
        ('curPolygonId', u16),
    ]
