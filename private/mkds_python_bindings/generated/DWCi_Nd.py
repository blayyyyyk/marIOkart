from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
DWC_ND_ERROR_NONE = 0
DWC_ND_ERROR_ALLOC = 1
DWC_ND_ERROR_BUSY = 2
DWC_ND_ERROR_HTTP = 3
DWC_ND_ERROR_BUFFULL = 4
DWC_ND_ERROR_DLSERVER = 5
DWC_ND_ERROR_CANCELED = 6
DWC_ND_ERROR_PARAM = 7
DWC_ND_ERROR_FATAL = 8
DWC_ND_ERROR_MAX = 9

DWC_ND_ERROR_NONE = 0
DWC_ND_ERROR_ALLOC = 1
DWC_ND_ERROR_BUSY = 2
DWC_ND_ERROR_HTTP = 3
DWC_ND_ERROR_BUFFULL = 4
DWC_ND_ERROR_DLSERVER = 5
DWC_ND_ERROR_CANCELED = 6
DWC_ND_ERROR_PARAM = 7
DWC_ND_ERROR_FATAL = 8
DWC_ND_ERROR_MAX = 9

DWC_ND_CBREASON_INITIALIZE = 0
DWC_ND_CBREASON_GETFILELISTNUM = 1
DWC_ND_CBREASON_GETFILELIST = 2
DWC_ND_CBREASON_GETFILE = 3
DWC_ND_CBREASON_MAX = 4

DWC_ND_CBREASON_INITIALIZE = 0
DWC_ND_CBREASON_GETFILELISTNUM = 1
DWC_ND_CBREASON_GETFILELIST = 2
DWC_ND_CBREASON_GETFILE = 3
DWC_ND_CBREASON_MAX = 4


class DWCNdFileInfo(Structure):
    _fields_ = [
        ('name', (c_char * 33)),
        ('explain', (c_short * 51)),
        ('param1', (c_char * 11)),
        ('param2', (c_char * 11)),
        ('param3', (c_char * 11)),
        ('size', c_int),
    ]
DWCNdAlloc = c_void_p32
DWCNdCallback = c_void_p32
DWCNdError = c_int
DWCNdFree = c_void_p32
DWCNdCleanupCallback = c_void_p32
DWCNdCallbackReason = c_int
