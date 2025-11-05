from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.mapobjInstance import *
from private.mkds_python_bindings.generated.pathwalker import *
from private.mkds_python_bindings.generated.spaEmitter import *

PAKKUN_FIRE_STATE_0 = 0
PAKKUN_FIRE_STATE_1 = 1

PAKKUN_FIRE_STATE_0 = 0
PAKKUN_FIRE_STATE_1 = 1


class pakkunfire_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('pathwalker', pw_pathwalker_t),
        ('elevation', fx32),
        ('elevationVelocity', fx32),
        ('state', PakkunFireState),
        ('emitter', POINTER32(spa_emitter_t)),
    ]
PakkunFireState = c_int
