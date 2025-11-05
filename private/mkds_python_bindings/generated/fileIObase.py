from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.types import *


class NNSMcsFile(Structure):
    _fields_ = [
        ('handle', u32),
        ('filesize', u32),
        ('errCode', u32),
        ('srvErrCode', u32),
        ('tempData', u32),
        ('bDone', u32),
    ]

class NNSMcsFileFindData(Structure):
    _fields_ = [
        ('attribute', u32),
        ('size', u32),
        ('name', (c_char * 260)),
    ]
