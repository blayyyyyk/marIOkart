from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.gpi import *
from private.mkds_python_bindings.generated.gpiBuffer import *
from private.mkds_python_bindings.generated.gsPlatform import *
from private.mkds_python_bindings.generated.gsPlatformSocket import *


class GPISearchData(Structure):
    _fields_ = [
        ('type', c_int),
        ('sock', SOCKET),
        ('inputBuffer', GPIBuffer),
        ('outputBuffer', GPIBuffer),
        ('nick', (c_char * 31)),
        ('uniquenick', (c_char * 21)),
        ('email', (c_char * 51)),
        ('firstname', (c_char * 31)),
        ('lastname', (c_char * 31)),
        ('password', (c_char * 31)),
        ('cdkey', (c_char * 65)),
        ('partnerID', c_int),
        ('icquin', c_int),
        ('skip', c_int),
        ('productID', c_int),
        ('processing', GPIBool),
        ('remove', GPIBool),
        ('searchStartTime', gsi_time),
        ('revBuddyProfileIds', POINTER32(c_int)),
        ('numOfRevBuddyProfiles', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_dwc_gs_gp_gpiSearch_h_42_9(Structure):
    _fields_ = [
        ('type', c_int),
        ('sock', SOCKET),
        ('inputBuffer', c_int),
        ('outputBuffer', c_int),
        ('nick', (c_char * 31)),
        ('uniquenick', (c_char * 21)),
        ('email', (c_char * 51)),
        ('firstname', (c_char * 31)),
        ('lastname', (c_char * 31)),
        ('password', (c_char * 31)),
        ('cdkey', (c_char * 65)),
        ('partnerID', c_int),
        ('icquin', c_int),
        ('skip', c_int),
        ('productID', c_int),
        ('processing', GPIBool),
        ('remove', GPIBool),
        ('searchStartTime', gsi_time),
        ('revBuddyProfileIds', POINTER32(c_int)),
        ('numOfRevBuddyProfiles', c_int),
    ]
