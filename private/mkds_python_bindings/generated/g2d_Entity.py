from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.g2d_Anim_data import *
from private.mkds_python_bindings.generated.g2d_Entity_data import *
from private.mkds_python_bindings.generated.g2d_PaletteTable import *
from private.mkds_python_bindings.generated.types import *


class NNSG2dEntity(Structure):
    _fields_ = [
        ('pDrawStuff', c_void_p32),
        ('pEntityData', POINTER32(NNSG2dEntityData)),
        ('pAnimDataBank', POINTER32(NNSG2dAnimBankData)),
        ('currentSequenceIdx', u16),
        ('pad16', u16),
        ('pPaletteTbl', POINTER32(NNSG2dPaletteSwapTable)),
    ]
