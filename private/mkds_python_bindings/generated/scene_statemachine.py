from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_scene_scene_statemachine_h_13_9(Structure):
    _fields_ = [
        ('frameCounter', c_int),
        ('changingState', c_int),
        ('prevState', c_int),
        ('curState', c_int),
        ('nextState', c_int),
    ]

class ssm_state_t(Structure):
    _fields_ = [
        ('vblankFunc', ssm_vblank_func_t),
        ('renderFunc', ssm_render_func_t),
    ]
ssm_vblank_func_t = c_void_p32
ssm_render_func_t = c_void_p32
