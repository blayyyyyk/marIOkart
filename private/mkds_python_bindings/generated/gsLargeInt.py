from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.gsPlatform import *


class gsLargeInt_s(Structure):
    _fields_ = [
        ('mLength', gsi_u32),
        ('mData', (gsi_u32 * 64)),
    ]
