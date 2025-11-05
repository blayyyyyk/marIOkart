from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.g2d_Font_data import *
from private.mkds_python_bindings.generated.g2di_SplitChar import *
from private.mkds_python_bindings.generated.types import *


class NNSG2dGlyph(Structure):
    _fields_ = [
        ('pWidths', POINTER32(NNSG2dCharWidths)),
        ('image', POINTER32(u8)),
    ]

class NNSG2dTextRect(Structure):
    _fields_ = [
        ('width', c_int),
        ('height', c_int),
    ]

class NNSG2dFont(Structure):
    _fields_ = [
        ('pRes', POINTER32(NNSG2dFontInformation)),
        ('cbCharSpliter', NNSiG2dSplitCharCallback),
        ('isOldVer', u16),
        ('widthsSize', u16),
    ]
