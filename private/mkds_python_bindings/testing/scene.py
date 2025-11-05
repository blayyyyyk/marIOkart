from ctypes import *
from private.mkds_python_bindings.testing.sceneProc import *
from private.mkds_python_bindings.testing.thread import *
from private.mkds_python_bindings.testing.types import *

SCENE_STATE_INITIALIZING = 0
SCENE_STATE_FADE_IN = 1
SCENE_STATE_RUNNING = 2
SCENE_STATE_FADE_OUT = 3
SCENE_STATE_FINALIZE_WAIT = 4
SCENE_STATE_FINALIZING = 5
SCENE_STATE_IDLE = 6

SCENE_STATE_INITIALIZING = 0
SCENE_STATE_FADE_IN = 1
SCENE_STATE_RUNNING = 2
SCENE_STATE_FADE_OUT = 3
SCENE_STATE_FINALIZE_WAIT = 4
SCENE_STATE_FINALIZING = 5
SCENE_STATE_IDLE = 6

SCENE_STATE_INITIALIZING = 0
SCENE_STATE_FADE_IN = 1
SCENE_STATE_RUNNING = 2
SCENE_STATE_FADE_OUT = 3
SCENE_STATE_FINALIZE_WAIT = 4
SCENE_STATE_FINALIZING = 5
SCENE_STATE_IDLE = 6

SCENE_STATE_INITIALIZING = 0
SCENE_STATE_FADE_IN = 1
SCENE_STATE_RUNNING = 2
SCENE_STATE_FADE_OUT = 3
SCENE_STATE_FINALIZE_WAIT = 4
SCENE_STATE_FINALIZING = 5
SCENE_STATE_IDLE = 6

SceneState = c_int
SceneState = c_int


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_scene_scene_h_15_9_(Structure):
    _fields_ = [
        ('threadStack', u32), #POINTER(u32)),
        ('thread', OSThread),
        ('threadQueue', OSThreadQueue),
        ('preSleepCallback', c_int),
        ('postSleepCallback', c_int),
        ('sceneFrameCounter', c_int),
        ('totalFrameCounter', c_int),
        ('curSceneDef', scene_def_t),
        ('state', SceneState),
        ('isLcdOff', BOOL),
        ('gap114', (u8 * 4)),
        ('field118', u32),
        ('field11C', u32),
        ('field120', s8),
        ('gap121', (u8 * 3)),
    ]
