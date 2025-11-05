from ctypes import *
from private.mkds_python_bindings.testing.types import *


class OSPrintWChar(Union):
    _fields_ = [
        ('s', u16),
        ('c', (c_char * 2)),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_os_common_printf_h_164_18_(Structure):
    _fields_ = [
        ('in', u32),
        ('out', u32),
        ('buffer', (OSPrintWChar * 512)),
    ]
