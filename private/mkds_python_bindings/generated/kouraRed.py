from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.sfx import *
from private.mkds_python_bindings.generated.types import *


class kourared_t(Structure):
    _fields_ = [
        ('gap0', (u8 * 600)),
        ('sfxExParams', sfx_emitter_ex_params_t),
    ]
