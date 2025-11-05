from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.types import *


class CPContext(Structure):
    _fields_ = [
        ('div_numer', u64),
        ('div_denom', u64),
        ('sqrt', u64),
        ('div_mode', u16),
        ('sqrt_mode', u16),
    ]

class OSContext(Structure):
    _fields_ = [
        ('cpsr', u32),
        ('r', (u32 * 13)),
        ('sp', u32),
        ('lr', u32),
        ('pc_plus4', u32),
        ('sp_svc', u32),
    ]
