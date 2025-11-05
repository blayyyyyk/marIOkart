from ctypes import *
from private.mkds_python_bindings.testing.types import *


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_oam_h_3_9_(Structure):
    _fields_ = [
        ('oam', (c_int * 128)),
        ('objCount', u16),
        ('affineCount', u16),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_oam_h_10_9_(Structure):
    _fields_ = [
        ('mainOamBuf', oam_buf_t),
        ('subOamBuf', oam_buf_t),
    ]
