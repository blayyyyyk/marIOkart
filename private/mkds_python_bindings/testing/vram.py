from ctypes import *
from private.mkds_python_bindings.testing.types import *

vram_wvr_load_unload_callback_func_t = u32


class vram_wvr_stat_t(Structure):
    _fields_ = [
        ('status', vu32),
        ('callbackFunc', vram_wvr_load_unload_callback_func_t),
    ]
