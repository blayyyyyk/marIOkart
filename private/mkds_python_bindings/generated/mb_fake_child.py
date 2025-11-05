from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.mb_gameinfo import *
from private.mkds_python_bindings.generated.types import *
from private.mkds_python_bindings.generated.wm import *

MB_FAKESCAN_PARENT_FOUND = 0
MB_FAKESCAN_PARENT_LOST = 1
MB_FAKESCAN_API_ERROR = 2
MB_FAKESCAN_END_SCAN = 3

MB_FAKESCAN_PARENT_FOUND = 0
MB_FAKESCAN_PARENT_LOST = 1
MB_FAKESCAN_API_ERROR = 2
MB_FAKESCAN_END_SCAN = 3


class MBFakeScanCallback(Structure):
    _fields_ = [
        ('index', u16),
        ('padding', u16),
        ('gameInfo', POINTER32(MBGameInfo)),
        ('bssDesc', POINTER32(WMBssDesc)),
    ]

class MBFakeScanErrorCallback(Structure):
    _fields_ = [
        ('apiid', u16),
        ('errcode', u16),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_mb_mb_fake_child_h_61_9(Structure):
    _fields_ = [
        ('index', u16),
        ('padding', u16),
        ('gameInfo', POINTER32(c_int)),
        ('bssDesc', POINTER32(WMBssDesc)),
    ]
MBFakeScanCallbackFunc = c_void_p32
