from ctypes import *

class cklcr_char_def_t(Structure):
    _fields_ = [
        ('playerModelName', u32), #POINTER(c_char)),
        ('enemyModelName', u32), #POINTER(c_char)),
        ('emblemName', u32), #POINTER(c_char)),
    ]

class cklcr_kart_def_t(Structure):
    _fields_ = [
        ('kartModelName', u32), #POINTER(c_char)),
        ('kartShadowName', u32), #POINTER(c_char)),
    ]
