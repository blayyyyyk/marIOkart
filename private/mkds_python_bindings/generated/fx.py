from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32

class struct__anonymous_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_fx_fx_h_276_5(Structure):
    _fields_ = [
        ('_00', fx32),
        ('_01', fx32),
        ('_02', fx32),
        ('_03', fx32),
        ('_10', fx32),
        ('_11', fx32),
        ('_12', fx32),
        ('_13', fx32),
        ('_20', fx32),
        ('_21', fx32),
        ('_22', fx32),
        ('_23', fx32),
        ('_30', fx32),
        ('_31', fx32),
        ('_32', fx32),
        ('_33', fx32),
    ]

class struct__anonymous_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_fx_fx_h_304_5(Structure):
    _fields_ = [
        ('_00', fx32),
        ('_01', fx32),
        ('_02', fx32),
        ('_10', fx32),
        ('_11', fx32),
        ('_12', fx32),
        ('_20', fx32),
        ('_21', fx32),
        ('_22', fx32),
        ('_30', fx32),
        ('_31', fx32),
        ('_32', fx32),
    ]

class struct__anonymous_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_fx_fx_h_331_5(Structure):
    _fields_ = [
        ('_00', fx32),
        ('_01', fx32),
        ('_02', fx32),
        ('_10', fx32),
        ('_11', fx32),
        ('_12', fx32),
        ('_20', fx32),
        ('_21', fx32),
        ('_22', fx32),
    ]

class struct__anonymous_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_fx_fx_h_356_5(Structure):
    _fields_ = [
        ('_00', fx32),
        ('_01', fx32),
        ('_10', fx32),
        ('_11', fx32),
    ]
fx64c = c_long
fx64 = c_long

class MtxFx44(Union):
    _fields_ = [
        ('m', ((fx32 * 4) * 4)),
        ('a', (fx32 * 16)),
    ]

class MtxFx22(Union):
    _fields_ = [
        ('m', ((fx32 * 2) * 2)),
        ('a', (fx32 * 4)),
    ]

class VecFx16(Structure):
    _fields_ = [
        ('x', fx16),
        ('y', fx16),
        ('z', fx16),
    ]

class VecFx32(Structure):
    _fields_ = [
        ('x', fx32),
        ('y', fx32),
        ('z', fx32),
    ]

class MtxFx43(Union):
    _fields_ = [
        ('m', ((fx32 * 3) * 4)),
        ('a', (fx32 * 12)),
    ]

class MtxFx33(Union):
    _fields_ = [
        ('m', ((fx32 * 3) * 3)),
        ('a', (fx32 * 9)),
    ]
fx32 = c_long
fx16 = c_short
