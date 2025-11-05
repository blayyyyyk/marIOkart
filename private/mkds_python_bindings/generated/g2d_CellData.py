from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.types import *


class NNSG2dCellData(Structure):
    _fields_ = [
        ('pOamAttrDataArray', POINTER32(NNSG2dCellOamAttrData)),
        ('numOamAttrData', u16),
        ('cellAttr', u16),
        ('posX', s16),
        ('posY', s16),
        ('posZ', s16),
    ]
