from ctypes import *
from private.mkds_python_bindings.testing.sfx import *
from private.mkds_python_bindings.testing.types import *


class kourawing_t(Structure):
    _fields_ = [
        ('gap0', (u8 * 768)),
        ('sfxExParams', sfx_emitter_ex_params_t),
    ]
