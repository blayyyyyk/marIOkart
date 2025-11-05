from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.jnLyt import *
from private.mkds_python_bindings.generated.jnLytRes import *
from private.mkds_python_bindings.generated.oam import *
from private.mkds_python_bindings.generated.scene_statemachine import *

TITLE_SCENE_BUTTON_SINGLE_PLAYER = 0
TITLE_SCENE_BUTTON_MULTIPLAYER = 1
TITLE_SCENE_BUTTON_NINTENDO_WFC = 2
TITLE_SCENE_BUTTON_OPTIONS = 3
TITLE_SCENE_BUTTON_RECORDS = 4

TITLE_SCENE_BUTTON_SINGLE_PLAYER = 0
TITLE_SCENE_BUTTON_MULTIPLAYER = 1
TITLE_SCENE_BUTTON_NINTENDO_WFC = 2
TITLE_SCENE_BUTTON_OPTIONS = 3
TITLE_SCENE_BUTTON_RECORDS = 4

TITLE_SCENE_STATE_FADE_IN = 0
TITLE_SCENE_STATE_BUTTONS_IN = 1
TITLE_SCENE_STATE_MAIN = 2
TITLE_SCENE_STATE_BUTTONS_OUT = 3
TITLE_SCENE_STATE_SELECTED_OUT_EFFECT = 4
TITLE_SCENE_STATE_GOTO_SELECTED = 5
TITLE_SCENE_STATE_GOTO_NICKNAME = 6

TITLE_SCENE_STATE_FADE_IN = 0
TITLE_SCENE_STATE_BUTTONS_IN = 1
TITLE_SCENE_STATE_MAIN = 2
TITLE_SCENE_STATE_BUTTONS_OUT = 3
TITLE_SCENE_STATE_SELECTED_OUT_EFFECT = 4
TITLE_SCENE_STATE_GOTO_SELECTED = 5
TITLE_SCENE_STATE_GOTO_NICKNAME = 6


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_scene_title_title_h_26_9(Structure):
    _fields_ = [
        ('stateMachine', ssm_t),
        ('states', (ssm_state_t * 7)),
        ('inactivityCounter', c_int),
        ('introFinished', c_int),
        ('selectedButton', TitleSceneButton),
        ('selectedBeforeActivation', c_int),
        ('bottomRowLeftRight', c_int),
        ('outEffectOffset', c_int),
        ('introFrame', c_int),
        ('layoutElements', POINTER32(jnui_layout_element_t)),
        ('layoutBncl', POINTER32(jnui_bncl_res_t)),
        ('layoutBnbl', POINTER32(jnui_bnbl_res_t)),
        ('layoutBnll', POINTER32(jnui_bnll_res_t)),
        ('mainPalette', POINTER32(c_int)),
        ('mainBgData', POINTER32(c_int)),
        ('subPalette', POINTER32(c_int)),
        ('subBgData', POINTER32(c_int)),
        ('mainScreenData', POINTER32(c_int)),
        ('mainScreenData2', POINTER32(c_int)),
        ('subScreenData', POINTER32(c_int)),
        ('padding', (c_int * 12)),
        ('subObjPalette', POINTER32(c_int)),
        ('subObjCharacterData', POINTER32(c_int)),
        ('cellDataBank', POINTER32(c_int)),
        ('padding2', c_int),
        ('mainOamBuf', oam_buf_t),
        ('subOamBuf', oam_buf_t),
        ('field8B8', c_int),
    ]
TitleSceneState = c_int
TitleSceneButton = c_int
