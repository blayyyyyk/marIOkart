from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.types import *


class DWCAuthWiFiId(Structure):
    _fields_ = [
        ('uId', u64),
        ('notAttestedId', u64),
        ('flg', u32),
    ]
