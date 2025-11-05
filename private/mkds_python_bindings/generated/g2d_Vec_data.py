from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.types import *


class struct__anonymous_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nnsys_g2d_fmt_g2d_Vec_data_h_65_5(Structure):
    _fields_ = [
        ('_00', fx32),
        ('_01', fx32),
        ('_10', fx32),
        ('_11', fx32),
        ('_20', fx32),
        ('_21', fx32),
    ]

class union__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nnsys_g2d_fmt_g2d_Vec_data_h_63_9(Union):
    _fields_ = [
        ('m', ((c_int * 2) * 3)),
        ('a', (c_int * 6)),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nnsys_g2d_fmt_g2d_Vec_data_h_65_5(Structure):
    _fields_ = [
        ('_00', c_int),
        ('_01', c_int),
        ('_10', c_int),
        ('_11', c_int),
        ('_20', c_int),
        ('_21', c_int),
    ]

class NNSG2dSVec2(Structure):
    _fields_ = [
        ('x', s16),
        ('y', s16),
    ]

class NNSG2dFVec2(Structure):
    _fields_ = [
        ('x', fx32),
        ('y', fx32),
    ]

class MtxFx32(Union):
    _fields_ = [
        ('m', ((fx32 * 2) * 3)),
        ('a', (fx32 * 6)),
    ]
