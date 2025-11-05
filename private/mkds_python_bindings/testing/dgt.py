from ctypes import *
MATHMD5Context = u32
MATHSHA1Context = u32


class DGTHash1Context(Structure):
    _fields_ = [
        ('length', c_long),
    ]

class union__anonymous_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_dgt_dgt_h_59_5_(Union):
    _fields_ = [
        ('state', (c_long * 4)),
    ]

class struct__anonymous_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_dgt_dgt_h_61_9_(Structure):
    _fields_ = [
        ('a', c_long),
        ('b', c_long),
        ('c', c_long),
        ('d', c_long),
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
