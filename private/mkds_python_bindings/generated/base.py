from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.list import *
from private.mkds_python_bindings.generated.types import *


class NNSMcsDeviceCaps(Structure):
    _fields_ = [
        ('deviceID', u32),
        ('maskResource', u32),
    ]

class NNSMcsRecvCBInfo(Structure):
    _fields_ = [
        ('channel', u32),
        ('cbFunc', NNSMcsRecvCallback),
        ('userData', u32),
        ('link', NNSFndLink),
    ]
NNSMcsRecvCallback = c_void_p32
