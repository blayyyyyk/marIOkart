from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_vram_h_10_9(Structure):
    _fields_ = [
        ('status', c_int),
        ('callbackFunc', vram_wvr_load_unload_callback_func_t),
    ]
vram_wvr_load_unload_callback_func_t = c_void_p32
