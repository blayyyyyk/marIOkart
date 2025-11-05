from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.mapobjInstance import *
from private.mkds_python_bindings.generated.pathwalker import *
from private.mkds_python_bindings.generated.types import *

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


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_mapobj_enemies_sun_h_14_9(Structure):
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
SunState = c_int
