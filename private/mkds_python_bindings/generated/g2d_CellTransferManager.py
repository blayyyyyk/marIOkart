from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.g2d_Image import *
from private.mkds_python_bindings.generated.types import *


class NNSG2dCellTransferState(Structure):
    _fields_ = [
        ('dstVramLocation', NNSG2dVRamLocation),
        ('szDst', u32),
        ('pSrcNCGR', c_void_p32),
        ('pSrcNCBR', c_void_p32),
        ('szSrcData', u32),
        ('bActive', BOOL),
        ('bDrawn', u32),
        ('bTransferRequested', u32),
        ('srcOffset', u32),
        ('szByte', u32),
    ]
VramTransferTaskRegisterFuncPtr = c_void_p32
