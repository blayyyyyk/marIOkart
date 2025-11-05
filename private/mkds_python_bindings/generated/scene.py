from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.sceneProc import *

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


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_scene_scene_h_15_9(Structure):
    _fields_ = [
        ('threadStack', POINTER32(c_int)),
        ('thread', c_int),
        ('threadQueue', c_int),
        ('preSleepCallback', c_int),
        ('postSleepCallback', c_int),
        ('sceneFrameCounter', c_int),
        ('totalFrameCounter', c_int),
        ('curSceneDef', scene_def_t),
        ('state', SceneState),
        ('isLcdOff', c_int),
        ('gap114', (c_int * 4)),
        ('field118', c_int),
        ('field11C', c_int),
        ('field120', c_int),
        ('gap121', (c_int * 3)),
    ]
SceneState = c_int
