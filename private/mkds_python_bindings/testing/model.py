from ctypes import *
from private.mkds_python_bindings.testing.nnsfnd import *
from private.mkds_python_bindings.testing.types import *


class model_res_t(Structure):
    _fields_ = [
        ('link', NNSFndLink),
        ('nsbmd', u32),
        ('texRes', u32), #POINTER(c_int)),
    ]

class model_t(Structure):
    _fields_ = [
        ('renderObj', c_int),
        ('cullReversed', u32),
        ('render1Mat1Shp', BOOL),
        ('res', model_res_t),
    ]
