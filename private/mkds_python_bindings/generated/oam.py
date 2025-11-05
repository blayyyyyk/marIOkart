from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_oam_h_3_9(Structure):
    _fields_ = [
        ('oam', (c_int * 128)),
        ('objCount', c_int),
        ('affineCount', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_oam_h_10_9(Structure):
    _fields_ = [
        ('mainOamBuf', oam_buf_t),
        ('subOamBuf', oam_buf_t),
    ]
