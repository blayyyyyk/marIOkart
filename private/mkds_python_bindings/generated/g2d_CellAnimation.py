from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.g2d_Anim_data import *
from private.mkds_python_bindings.generated.g2d_Animation import *
from private.mkds_python_bindings.generated.g2d_Cell_data import *
from private.mkds_python_bindings.generated.g2d_SRTControl_data import *
from private.mkds_python_bindings.generated.types import *

NNSG2dCellAnimSequence = NNSG2dAnimSequenceData

class NNSG2dCellAnimation(Structure):
    _fields_ = [
        ('animCtrl', NNSG2dAnimController),
        ('pCurrentCell', POINTER32(NNSG2dCellData)),
        ('pCellDataBank', POINTER32(NNSG2dCellDataBank)),
        ('cellTransferStateHandle', u32),
        ('srtCtrl', NNSG2dSRTControl),
    ]
NNSG2dCellAnimBankData = NNSG2dAnimBankData
