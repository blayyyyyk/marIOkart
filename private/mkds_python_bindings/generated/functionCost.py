from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.types import *


class OSFunctionCostInfo(Structure):
    _fields_ = [
        ('current', POINTER32(OSFunctionCost)),
        ('limit', POINTER32(OSFunctionCost)),
        ('enable', u16),
        ('padding', u16),
        ('array', (OSFunctionCost * 1)),
    ]

class OSFunctionCostStatisticsInfo(Structure):
    _fields_ = [
        ('size', u32),
        ('limit', POINTER32(OSFunctionCostStatistics)),
        ('array', (OSFunctionCostStatistics * 1)),
    ]

class union__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_os_common_functionCost_h_72_9(Union):
    _fields_ = [
        ('entry', struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_os_common_functionCost_h_74_5),
        ('exit', struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_os_common_functionCost_h_83_5),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_os_common_functionCost_h_98_9(Structure):
    _fields_ = [
        ('current', POINTER32(OSFunctionCost)),
        ('limit', POINTER32(OSFunctionCost)),
        ('enable', c_int),
        ('padding', c_int),
        ('array', (OSFunctionCost * 1)),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_os_common_functionCost_h_115_9(Structure):
    _fields_ = [
        ('name', c_int),
        ('count', c_int),
        ('time', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_os_common_functionCost_h_123_9(Structure):
    _fields_ = [
        ('size', c_int),
        ('limit', POINTER32(OSFunctionCostStatistics)),
        ('array', (OSFunctionCostStatistics * 1)),
    ]

class OSFunctionCost(Union):
    _fields_ = [
        ('entry', struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_os_common_functionCost_h_74_5),
        ('exit', struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_os_common_functionCost_h_83_5),
    ]

class OSFunctionCostStatistics(Structure):
    _fields_ = [
        ('name', u32),
        ('count', u32),
        ('time', u64),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_os_common_functionCost_h_74_5(Structure):
    _fields_ = [
        ('name', c_int),
        ('time', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_os_common_functionCost_h_83_5(Structure):
    _fields_ = [
        ('tag', c_int),
        ('time', c_int),
    ]
