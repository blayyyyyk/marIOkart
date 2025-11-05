from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32

class _GOACryptState(Structure):
    _fields_ = [
        ('cards', (c_char * 256)),
        ('rotor', c_char),
        ('ratchet', c_char),
        ('avalanche', c_char),
        ('last_plain', c_char),
        ('last_cipher', c_char),
    ]
