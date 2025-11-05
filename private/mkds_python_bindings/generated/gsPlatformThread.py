from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_dwc_gs_common_gsPlatformThread_h_55_10(Structure):
    _fields_ = [
        ('mThread', c_int),
        ('mStack', c_void_p32),
    ]
GSISemaphoreID = c_int
GSThreadFunc = c_void_p32
GSICriticalSection = c_int
