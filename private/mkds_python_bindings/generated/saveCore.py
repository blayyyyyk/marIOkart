from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
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


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_save_saveCore_h_32_9(Structure):
    _fields_ = [
        ('status', SaveCoreStatus),
        ('error', SaveCoreError),
        ('backupLock', c_int),
        ('isEnabled', c_int),
        ('isBusy', c_int),
        ('transferType', SaveCoreTransferType),
        ('backupSrcDst', c_int),
        ('originalError', SaveCoreError),
        ('originalSrcDst', POINTER32(c_int)),
        ('originalTimestamp', c_int),
        ('field26', c_int),
        ('readDst', POINTER32(c_int)),
        ('readSrc', c_int),
        ('readLen', c_int),
        ('readBlockSignature', c_int),
        ('readBlockIsHeader', c_int),
        ('field3C', c_int),
        ('writeSrc', POINTER32(c_int)),
        ('writeDst', c_int),
        ('writeLength', c_int),
        ('field4C', c_int),
        ('writeBlockIsHeader', c_int),
        ('callbackArg', c_void_p32),
        ('field58', c_int),
        ('field5C', c_int),
        ('tmpBuf', c_void_p32),
        ('testByte', c_void_p32),
        ('realDst', c_int),
        ('field6C', c_int),
        ('field6D', c_int),
        ('field6E', c_int),
        ('field6F', c_int),
    ]
savc_transfer_callback_t = c_void_p32
SaveCoreTransferType = c_int
SaveCoreError = c_int
SaveCoreStatus = c_int
