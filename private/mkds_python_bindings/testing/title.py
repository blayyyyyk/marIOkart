from ctypes import *
from private.mkds_python_bindings.testing.jnLyt import *
from private.mkds_python_bindings.testing.jnLytRes import *
from private.mkds_python_bindings.testing.oam import *
from private.mkds_python_bindings.testing.scene_statemachine import *
from private.mkds_python_bindings.testing.types import *

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

TitleSceneButton = c_int
TitleSceneState = c_int


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_scene_title_title_h_26_9_(Structure):
    _fields_ = [
        ('stateMachine', ssm_t),
        ('states', (ssm_state_t * 7)),
        ('inactivityCounter', c_int),
        ('introFinished', u32),
        ('selectedButton', TitleSceneButton),
        ('selectedBeforeActivation', BOOL),
        ('bottomRowLeftRight', u32),
        ('outEffectOffset', s16),
        ('introFrame', u32),
        ('layoutElements', u32), #POINTER(jnui_layout_element_t)),
        ('layoutBncl', u32), #POINTER(jnui_bncl_res_t)),
        ('layoutBnbl', u32), #POINTER(jnui_bnbl_res_t)),
        ('layoutBnll', u32), #POINTER(jnui_bnll_res_t)),
        ('mainPalette', u32), #POINTER(c_int)),
        ('mainBgData', u32), #POINTER(c_int)),
        ('subPalette', u32), #POINTER(c_int)),
        ('subBgData', u32), #POINTER(c_int)),
        ('mainScreenData', u32), #POINTER(c_int)),
        ('mainScreenData2', u32), #POINTER(c_int)),
        ('subScreenData', u32), #POINTER(c_int)),
        ('padding', (u8 * 12)),
        ('subObjPalette', u32), #POINTER(c_int)),
        ('subObjCharacterData', u32), #POINTER(c_int)),
        ('cellDataBank', u32), #POINTER(c_int)),
        ('padding2', u32),
        ('mainOamBuf', oam_buf_t),
        ('subOamBuf', oam_buf_t),
        ('field8B8', u32),
    ]
