from ctypes import *
from private.mkds_python_bindings.testing.animationManager import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.model import *
from private.mkds_python_bindings.testing.sinThing import *
from private.mkds_python_bindings.testing.spaEmitter import *
from private.mkds_python_bindings.testing.types import *

TROPHY_ID_KINO = 0
TROPHY_ID_FLOW = 1
TROPHY_ID_STAR = 2
TROPHY_ID_SPEC = 3
TROPHY_ID_KORA = 4
TROPHY_ID_BANA = 5
TROPHY_ID_KONO = 6
TROPHY_ID_THUN = 7
TROPHY_ID_GENERIC = 8
TROPHY_ID_MAX = 9

TROPHY_ID_KINO = 0
TROPHY_ID_FLOW = 1
TROPHY_ID_STAR = 2
TROPHY_ID_SPEC = 3
TROPHY_ID_KORA = 4
TROPHY_ID_BANA = 5
TROPHY_ID_KONO = 6
TROPHY_ID_THUN = 7
TROPHY_ID_GENERIC = 8
TROPHY_ID_MAX = 9

TROPHY_RANK_GOLD = 0
TROPHY_RANK_SILVER = 1
TROPHY_RANK_BRONZE = 2
TROPHY_RANK_MAX = 3

TROPHY_RANK_GOLD = 0
TROPHY_RANK_SILVER = 1
TROPHY_RANK_BRONZE = 2
TROPHY_RANK_MAX = 3

TrophyId = c_int
TrophyRank = c_int


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_scene_result_trophy_h_31_9_(Structure):
    _fields_ = [
        ('trophyId', u16),
        ('trophyRank', u16),
        ('rotY', u16),
        ('rotZ', sinthing_t),
        ('trophyModel', model_t),
        ('trophyNsbtpAnim', anim_manager_t),
        ('identity', MtxFx43),
        ('light0DirX', fx16),
        ('light1DirX', fx16),
        ('light0DirY', fx16),
        ('light1DirY', fx16),
        ('light0DirZ', fx16),
        ('light1DirZ', fx16),
        ('sparkleEmitterPos', VecFx32),
        ('sparkleEmitterRadius', fx32),
        ('sparkleEmitter', u32), #POINTER(spa_emitter_t)),
        ('goldConfettiEmitter', u32), #POINTER(spa_emitter_t)),
        ('silverConfettiEmitter', u32), #POINTER(spa_emitter_t)),
    ]
