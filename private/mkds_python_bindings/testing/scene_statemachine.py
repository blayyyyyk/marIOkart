from ctypes import *
from private.mkds_python_bindings.testing.types import *

ssm_render_func_t = u32
ssm_render_func_t = u32
ssm_vblank_func_t = u32
ssm_vblank_func_t = u32


class ssm_state_t(Structure):
    _fields_ = [
        ('vblankFunc', ssm_vblank_func_t),
        ('renderFunc', ssm_render_func_t),
    ]

class ssm_t(Structure):
    _fields_ = [
        ('frameCounter', c_int),
        ('changingState', BOOL),
        ('prevState', u32),
        ('curState', u32),
        ('nextState', u32),
    ]
