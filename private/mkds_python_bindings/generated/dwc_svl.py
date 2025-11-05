from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.types import *

DWC_SVL_STATE_DIRTY = 0
DWC_SVL_STATE_IDLE = 1
DWC_SVL_STATE_HTTP = 2
DWC_SVL_STATE_SUCCESS = 3
DWC_SVL_STATE_ERROR = 4
DWC_SVL_STATE_CANCELED = 5
DWC_SVL_STATE_MAX = 6

DWC_SVL_STATE_DIRTY = 0
DWC_SVL_STATE_IDLE = 1
DWC_SVL_STATE_HTTP = 2
DWC_SVL_STATE_SUCCESS = 3
DWC_SVL_STATE_ERROR = 4
DWC_SVL_STATE_CANCELED = 5
DWC_SVL_STATE_MAX = 6


class DWCSvlResult(Structure):
    _fields_ = [
        ('status', BOOL),
        ('svlhost', (c_char * 65)),
        ('svltoken', (c_char * 301)),
    ]
DWCSvlState = c_int
