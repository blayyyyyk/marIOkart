from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.res_struct import *


class NNSG3dUtilResName(Union):
    _fields_ = [
        ('_0', (c_char * 16)),
    ]

class struct__anonymous_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nnsys_g3d_util_h_90_5(Structure):
    _fields_ = [
        ('resName', NNSG3dResName),
    ]
