from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32

class cklcr_char_def_t(Structure):
    _fields_ = [
        ('playerModelName', POINTER32(c_char)),
        ('enemyModelName', POINTER32(c_char)),
        ('emblemName', POINTER32(c_char)),
    ]

class cklcr_kart_def_t(Structure):
    _fields_ = [
        ('kartModelName', POINTER32(c_char)),
        ('kartShadowName', POINTER32(c_char)),
    ]
