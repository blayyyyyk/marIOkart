from ctypes import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.mapobjInstance import *
from private.mkds_python_bindings.testing.pathwalker import *
from private.mkds_python_bindings.testing.quaternion import *
from private.mkds_python_bindings.testing.types import *

KURIBO_STATE_ROUTE_IDLE = 0
KURIBO_STATE_GROUND_ROAM = 1
KURIBO_STATE_HIT_GROW = 2
KURIBO_STATE_HIT_SHRINK = 3
KURIBO_STATE_DEAD = 4
KURIBO_STATE_REAPPEAR = 5

KURIBO_STATE_ROUTE_IDLE = 0
KURIBO_STATE_GROUND_ROAM = 1
KURIBO_STATE_HIT_GROW = 2
KURIBO_STATE_HIT_SHRINK = 3
KURIBO_STATE_DEAD = 4
KURIBO_STATE_REAPPEAR = 5

KuriboState = c_int


class kuribo_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('fieldA0', fx32),
        ('direction', quaternion_t),
        ('targetDir', quaternion_t),
        ('squashRatio', fx32),
        ('squashVelocity', fx32),
        ('pathWalker', pw_pathwalker_t),
        ('frame', u16),
        ('fieldF4', BOOL),
        ('dirInterpRatio', u16),
        ('reappearAfterHit', BOOL),
        ('alpha', u16),
        ('field102', s16),
        ('field104', u32),
    ]
