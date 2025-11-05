from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.types import *


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_os_common_printf_h_164_18(Structure):
    _fields_ = [
        ('in', u32),
        ('out', u32),
        ('buffer', (OSPrintWChar * 512)),
    ]

class OSPrintWChar(Union):
    _fields_ = [
        ('s', u16),
        ('c', (c_char * 2)),
    ]
