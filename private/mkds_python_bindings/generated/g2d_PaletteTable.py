from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.types import *


class NNSG2dPaletteSwapTable(Structure):
    _fields_ = [
        ('paletteIndex', (u16 * 16)),
    ]
