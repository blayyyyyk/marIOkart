from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.mapobjInstance import *
from private.mkds_python_bindings.generated.pathwalker import *
from private.mkds_python_bindings.generated.types import *

DOSSUN_STATE_0 = 0
DOSSUN_STATE_1 = 1
DOSSUN_STATE_2 = 2
DOSSUN_STATE_3 = 3
DOSSUN_STATE_4 = 4
DOSSUN_STATE_5 = 5
DOSSUN_STATE_6 = 6

DOSSUN_STATE_0 = 0
DOSSUN_STATE_1 = 1
DOSSUN_STATE_2 = 2
DOSSUN_STATE_3 = 3
DOSSUN_STATE_4 = 4
DOSSUN_STATE_5 = 5
DOSSUN_STATE_6 = 6

DOSSUN_STAR_HIT_ANIM_STATE_INACTIVE = 0
DOSSUN_STAR_HIT_ANIM_STATE_1 = 1
DOSSUN_STAR_HIT_ANIM_STATE_2 = 2

DOSSUN_STAR_HIT_ANIM_STATE_INACTIVE = 0
DOSSUN_STAR_HIT_ANIM_STATE_1 = 1
DOSSUN_STAR_HIT_ANIM_STATE_2 = 2


class dossun_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('state', DossunState),
        ('stateCounter', c_int),
        ('someSpeed', fx32),
        ('floorY', fx32),
        ('isSmashing', BOOL),
        ('starHitAnimState', DossunStarHitAnimState),
        ('rotYDelta', s16),
        ('rotY', s16),
        ('lastStarHitFrame', u32),
        ('noStarHitPlayerMask', u16),
        ('sinAng', u16),
        ('sinAmplitude', fx32),
        ('pathwalker', pw_pathwalker_t),
        ('initialPathPoint', u16),
        ('isHorizontalMoveType', BOOL),
        ('fieldF4', VecFx32),
        ('field100', fx32),
        ('someAcceleration', fx32),
        ('anotherSpeed', fx32),
    ]
DossunStarHitAnimState = c_int
DossunState = c_int
