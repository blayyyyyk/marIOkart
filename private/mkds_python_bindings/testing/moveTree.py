from ctypes import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.mapobjInstance import *
from private.mkds_python_bindings.testing.objectShadow import *
from private.mkds_python_bindings.testing.pathwalker import *
from private.mkds_python_bindings.testing.types import *

MTREE_STATE_WAIT = 0
MTREE_STATE_WALK = 1

MTREE_STATE_WAIT = 0
MTREE_STATE_WALK = 1

MTreeState = c_int


class movetree_t(Structure):
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
