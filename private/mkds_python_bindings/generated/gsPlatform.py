from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32

class tm(Structure):
    _fields_ = [
        ('tm_sec', c_int),
        ('tm_min', c_int),
        ('tm_hour', c_int),
        ('tm_mday', c_int),
        ('tm_mon', c_int),
        ('tm_year', c_int),
        ('tm_wday', c_int),
        ('tm_yday', c_int),
        ('tm_isdst', c_int),
    ]
gsi_u64 = c_long
gsi_i8 = c_char
gsi_i64 = c_long
goa_int32 = c_int
goa_uint32 = c_int
gsi_u32 = c_int
gsi_i16 = c_short
time_t = c_int
gsi_u16 = c_short
gsi_bool = c_int
gsi_i32 = c_int
gsi_u8 = c_char
gsi_time = c_int
