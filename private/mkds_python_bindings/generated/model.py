from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.kernel import *
from private.mkds_python_bindings.generated.list import *
from private.mkds_python_bindings.generated.res_struct import *
from private.mkds_python_bindings.generated.types import *


class model_res_t(Structure):
    _fields_ = [
        ('link', NNSFndLink),
        ('nsbmd', c_void_p32),
        ('texRes', POINTER32(NNSG3dResTex)),
    ]

class model_t(Structure):
    _fields_ = [
        ('renderObj', NNSG3dRenderObj),
        ('cullReversed', u32),
        ('render1Mat1Shp', BOOL),
        ('res', model_res_t),
    ]
