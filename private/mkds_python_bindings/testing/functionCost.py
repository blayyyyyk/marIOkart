from ctypes import *
from private.mkds_python_bindings.testing.types import *


class OSFunctionCost(Union):
    _fields_ = [
        ('entry', struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_os_common_functionCost_h_74_5_),
        ('exit', struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_os_common_functionCost_h_83_5_),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_os_common_functionCost_h_74_5_(Structure):
    _fields_ = [
        ('name', u32),
        ('time', u32),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_os_common_functionCost_h_83_5_(Structure):
    _fields_ = [
        ('tag', u32),
        ('time', u32),
    ]

class OSFunctionCostInfo(Structure):
    _fields_ = [
        ('current', u32), #POINTER(OSFunctionCost)),
        ('limit', u32), #POINTER(OSFunctionCost)),
        ('enable', u16),
        ('padding', u16),
        ('array', (OSFunctionCost * 1)),
    ]

class OSFunctionCostStatistics(Structure):
    _fields_ = [
        ('name', u32),
        ('count', u32),
        ('time', u64),
    ]

class OSFunctionCostStatisticsInfo(Structure):
    _fields_ = [
        ('size', u32),
        ('limit', u32), #POINTER(OSFunctionCostStatistics)),
        ('array', (OSFunctionCostStatistics * 1)),
    ]
