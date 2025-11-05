from ctypes import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.mapobjInstance import *
from private.mkds_python_bindings.testing.pathwalker import *
from private.mkds_python_bindings.testing.types import *

SUN_STATE_SLEEP = 0
SUN_STATE_MOVE = 1
SUN_STATE_WAIT = 2
SUN_STATE_SPAWN_SNAKE = 3
SUN_STATE_SWIRL = 4

SUN_STATE_SLEEP = 0
SUN_STATE_MOVE = 1
SUN_STATE_WAIT = 2
SUN_STATE_SPAWN_SNAKE = 3
SUN_STATE_SWIRL = 4

SunState = c_int


class sun_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
        ('fireSnakeSpawnCount', u32),
        ('fireSnakeSpawnRotY', u16),
        ('waitCounter', s32),
        ('rotZA', u32),
        ('rotZB', u16),
        ('nsbcaFrame', u16),
        ('nsbcaAnimSpeed', fx32),
        ('pathwalker', pw_pathwalker_t),
        ('state', SunState),
    ]
