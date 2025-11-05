from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
MATHMD5Context = DGTHash1Context
MATHSHA1Context = DGTHash2Context

class union__anonymous_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_dgt_dgt_h_59_5(Union):
    _fields_ = [
        ('state', (c_long * 4)),
    ]

class struct__anonymous_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_dgt_dgt_h_61_9(Structure):
    _fields_ = [
        ('a', c_long),
        ('b', c_long),
        ('c', c_long),
        ('d', c_long),
    ]

class DGTHash1Context(Structure):
    _fields_ = [
        ('length', c_long),
    ]

class DGTHash2Context(Structure):
    _fields_ = [
        ('h0', c_long),
        ('h1', c_long),
        ('h2', c_long),
        ('h3', c_long),
        ('h4', c_long),
        ('Nl', c_long),
        ('Nh', c_long),
        ('num', c_int),
        ('data', (c_long * 16)),
        ('dummy', (c_int * 2)),
    ]
