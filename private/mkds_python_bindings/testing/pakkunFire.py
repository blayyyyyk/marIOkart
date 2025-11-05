from ctypes import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.mapobjInstance import *
from private.mkds_python_bindings.testing.pathwalker import *
from private.mkds_python_bindings.testing.spaEmitter import *

PAKKUN_FIRE_STATE_0 = 0
PAKKUN_FIRE_STATE_1 = 1

PAKKUN_FIRE_STATE_0 = 0
PAKKUN_FIRE_STATE_1 = 1

PakkunFireState = c_int


class pakkunfire_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('pathwalker', pw_pathwalker_t),
        ('elevation', fx32),
        ('elevationVelocity', fx32),
        ('state', PakkunFireState),
        ('emitter', u32), #POINTER(spa_emitter_t)),
    ]
