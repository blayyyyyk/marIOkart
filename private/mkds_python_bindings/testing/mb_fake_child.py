from ctypes import *
from private.mkds_python_bindings.testing.types import *
from private.mkds_python_bindings.testing.wm import *

MB_FAKESCAN_PARENT_FOUND = 0
MB_FAKESCAN_PARENT_LOST = 1
MB_FAKESCAN_API_ERROR = 2
MB_FAKESCAN_END_SCAN = 3

MBFakeScanCallbackFunc = u32


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_mb_mb_fake_child_h_61_9_(Structure):
    _fields_ = [
        ('index', u16),
        ('padding', u16),
        ('gameInfo', u32), #POINTER(c_int)),
        ('bssDesc', u32), #POINTER(WMBssDesc)),
    ]

class MBFakeScanErrorCallback(Structure):
    _fields_ = [
        ('apiid', u16),
        ('errcode', u16),
    ]
