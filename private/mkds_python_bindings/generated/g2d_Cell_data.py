from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.g2d_Character_data import *
from private.mkds_python_bindings.generated.g2d_Common_data import *
from private.mkds_python_bindings.generated.types import *


class NNSG2dCellDataWithBR(Structure):
    _fields_ = [
        ('cellData', NNSG2dCellData),
        ('boundingRect', NNSG2dCellBoundingRectS16),
    ]

class NNSG2dCellDataBankBlock(Structure):
    _fields_ = [
        ('blockHeader', NNSG2dBinaryBlockHeader),
        ('cellDataBank', NNSG2dCellDataBank),
    ]

class NNSG2dUserExCellAttrBank(Structure):
    _fields_ = [
        ('numCells', u16),
        ('numAttribute', u16),
        ('pCellAttrArray', POINTER32(NNSG2dUserExCellAttr)),
    ]

class NNSG2dCellBoundingRectS16(Structure):
    _fields_ = [
        ('maxX', s16),
        ('maxY', s16),
        ('minX', s16),
        ('minY', s16),
    ]

class NNSG2dUserExCellAttr(Structure):
    _fields_ = [
        ('pAttr', POINTER32(u32)),
    ]

class NNSG2dCellDataBank(Structure):
    _fields_ = [
        ('numCells', u16),
        ('cellBankAttr', u16),
        ('pCellDataArrayHead', POINTER32(NNSG2dCellData)),
        ('mappingMode', NNSG2dCharacterDataMapingType),
        ('pVramTransferData', POINTER32(NNSG2dVramTransferData)),
        ('pStringBank', c_void_p32),
        ('pExtendedData', c_void_p32),
    ]

class NNSG2dVramTransferData(Structure):
    _fields_ = [
        ('szByteMax', u32),
        ('pCellTransferDataArray', POINTER32(NNSG2dCellVramTransferData)),
    ]

class NNSG2dCellData(Structure):
    _fields_ = [
        ('numOAMAttrs', u16),
        ('cellAttr', u16),
        ('pOamAttrArray', POINTER32(NNSG2dCellOAMAttrData)),
    ]

class NNSG2dCellVramTransferData(Structure):
    _fields_ = [
        ('srcDataOffset', u32),
        ('szByte', u32),
    ]

class NNSG2dCellOAMAttrData(Structure):
    _fields_ = [
        ('attr0', u16),
        ('attr1', u16),
        ('attr2', u16),
    ]
