from ctypes import *
from private.mkds_python_bindings.testing.types import *

SAVC_STATUS_0 = 0
SAVC_STATUS_1 = 1
SAVC_STATUS_2 = 2
SAVC_STATUS_3 = 3
SAVC_STATUS_4 = 4

SAVC_STATUS_0 = 0
SAVC_STATUS_1 = 1
SAVC_STATUS_2 = 2
SAVC_STATUS_3 = 3
SAVC_STATUS_4 = 4

SAVC_ERROR_NONE = 0
SAVC_ERROR_1 = 1
SAVC_ERROR_2 = 2
SAVC_ERROR_INVALID_BLOCK = 3
SAVC_ERROR_INVALID_HEADER_SIGNATURE = 4

SAVC_ERROR_NONE = 0
SAVC_ERROR_1 = 1
SAVC_ERROR_2 = 2
SAVC_ERROR_INVALID_BLOCK = 3
SAVC_ERROR_INVALID_HEADER_SIGNATURE = 4

SAVC_TRANSFER_TYPE_DIRECT = 0
SAVC_TRANSFER_TYPE_WITH_BACKUP = 1
SAVC_TRANSFER_TYPE_WITH_BACKUP_RETRY = 2

SAVC_TRANSFER_TYPE_DIRECT = 0
SAVC_TRANSFER_TYPE_WITH_BACKUP = 1
SAVC_TRANSFER_TYPE_WITH_BACKUP_RETRY = 2

SAVC_STATUS_0 = 0
SAVC_STATUS_1 = 1
SAVC_STATUS_2 = 2
SAVC_STATUS_3 = 3
SAVC_STATUS_4 = 4

SAVC_STATUS_0 = 0
SAVC_STATUS_1 = 1
SAVC_STATUS_2 = 2
SAVC_STATUS_3 = 3
SAVC_STATUS_4 = 4

SAVC_ERROR_NONE = 0
SAVC_ERROR_1 = 1
SAVC_ERROR_2 = 2
SAVC_ERROR_INVALID_BLOCK = 3
SAVC_ERROR_INVALID_HEADER_SIGNATURE = 4

SAVC_ERROR_NONE = 0
SAVC_ERROR_1 = 1
SAVC_ERROR_2 = 2
SAVC_ERROR_INVALID_BLOCK = 3
SAVC_ERROR_INVALID_HEADER_SIGNATURE = 4

SAVC_TRANSFER_TYPE_DIRECT = 0
SAVC_TRANSFER_TYPE_WITH_BACKUP = 1
SAVC_TRANSFER_TYPE_WITH_BACKUP_RETRY = 2

SAVC_TRANSFER_TYPE_DIRECT = 0
SAVC_TRANSFER_TYPE_WITH_BACKUP = 1
SAVC_TRANSFER_TYPE_WITH_BACKUP_RETRY = 2

SaveCoreError = c_int
SaveCoreError = c_int
SaveCoreStatus = c_int
SaveCoreStatus = c_int
SaveCoreTransferType = c_int
SaveCoreTransferType = c_int
savc_transfer_callback_t = u32
savc_transfer_callback_t = u32


class save_core_t(Structure):
    _fields_ = [
        ('status', SaveCoreStatus),
        ('error', SaveCoreError),
        ('backupLock', u32),
        ('isEnabled', BOOL),
        ('isBusy', BOOL),
        ('transferType', SaveCoreTransferType),
        ('backupSrcDst', u32),
        ('originalError', SaveCoreError),
        ('originalSrcDst', u32), #POINTER(u8)),
        ('originalTimestamp', u16),
        ('field26', u16),
        ('readDst', u32), #POINTER(u8)),
        ('readSrc', u32),
        ('readLen', u32),
        ('readBlockSignature', u32),
        ('readBlockIsHeader', BOOL),
        ('field3C', u32),
        ('writeSrc', u32), #POINTER(u8)),
        ('writeDst', u32),
        ('writeLength', u32),
        ('field4C', u32),
        ('writeBlockIsHeader', BOOL),
        ('callbackArg', u32),
        ('field58', u32),
        ('field5C', u32),
        ('tmpBuf', u32),
        ('testByte', u32),
        ('realDst', u32),
        ('field6C', u8),
        ('field6D', u8),
        ('field6E', u8),
        ('field6F', u8),
    ]
