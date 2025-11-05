from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.gp import *

pGPIProfile = c_void_p32

class GPIInfoCache(Structure):
    _fields_ = [
        ('nick', POINTER32(c_char)),
        ('uniquenick', POINTER32(c_char)),
        ('email', POINTER32(c_char)),
        ('firstname', POINTER32(c_char)),
        ('lastname', POINTER32(c_char)),
        ('homepage', POINTER32(c_char)),
        ('icquin', c_int),
        ('zipcode', (c_char * 11)),
        ('countrycode', (c_char * 3)),
        ('longitude', c_float),
        ('latitude', c_float),
        ('place', (c_char * 128)),
        ('birthday', c_int),
        ('birthmonth', c_int),
        ('birthyear', c_int),
        ('sex', GPEnum),
        ('publicmask', c_int),
        ('aimname', POINTER32(c_char)),
        ('pic', c_int),
        ('occupationid', c_int),
        ('industryid', c_int),
        ('incomeid', c_int),
        ('marriedid', c_int),
        ('childcount', c_int),
        ('interests1', c_int),
        ('ownership1', c_int),
        ('conntypeid', c_int),
    ]
