from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.gp import *
from private.mkds_python_bindings.generated.gpi import *
from private.mkds_python_bindings.generated.gpiCallback import *


class GPIOperation_s(Structure):
    _fields_ = [
        ('type', c_int),
        ('data', c_void_p32),
        ('blocking', GPIBool),
        ('callback', GPICallback),
        ('state', c_int),
        ('id', c_int),
        ('result', GPResult),
        ('pnext', POINTER32(GPIOperation_s)),
    ]

class GPIConnectData(Structure):
    _fields_ = [
        ('serverChallenge', (c_char * 128)),
        ('userChallenge', (c_char * 33)),
        ('passwordHash', (c_char * 33)),
        ('authtoken', (c_char * 256)),
        ('partnerchallenge', (c_char * 256)),
        ('cdkey', (c_char * 65)),
        ('newuser', GPIBool),
    ]
