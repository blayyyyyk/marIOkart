from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.mapobjInstance import *
from private.mkds_python_bindings.generated.objectShadow import *
from private.mkds_python_bindings.generated.pathwalker import *
from private.mkds_python_bindings.generated.types import *

MTREE_STATE_WAIT = 0
MTREE_STATE_WALK = 1

MTREE_STATE_WAIT = 0
MTREE_STATE_WALK = 1


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_mapobj_enemies_moveTree_h_12_9(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('pointDuration', s32),
        ('counter', s32),
        ('speed', fx32),
        ('nsbcaFrame', u16),
        ('nsbcaFrameDelta', s32),
        ('pathwalker', pw_pathwalker_t),
        ('state', MTreeState),
        ('shadow', objshadow_t),
    ]
MTreeState = c_int
