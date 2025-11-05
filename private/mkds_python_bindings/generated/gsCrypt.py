from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.gsLargeInt import *
from private.mkds_python_bindings.generated.gsPlatform import *


class gsCryptRSAOAEPPacket(Structure):
    _fields_ = [
        ('headerByte', gsi_u8),
        ('maskedSeed', (gsi_u8 * 20)),
        ('maskedData', (gsi_u8 * 107)),
    ]

class gsCryptRSAPKCS1Packet(Structure):
    _fields_ = [
        ('headerByte', (gsi_u8 * 2)),
        ('data', (gsi_u8 * 126)),
    ]

class gsCryptRSAKey(Structure):
    _fields_ = [
        ('modulus', gsLargeInt_t),
        ('exponent', gsLargeInt_t),
    ]
