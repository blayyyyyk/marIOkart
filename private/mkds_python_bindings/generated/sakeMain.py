from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.gsPlatform import *
from private.mkds_python_bindings.generated.sake import *


class SAKEInternal(Structure):
    _fields_ = [
        ('mIsGameAuthenticated', gsi_bool),
        ('mGameName', (c_char * 16)),
        ('mGameId', c_int),
        ('mSecretKey', (c_char * 9)),
        ('mIsProfileAuthenticated', gsi_bool),
        ('mProfileId', c_int),
        ('mLoginTicket', (c_char * 25)),
        ('mStartRequestResult', SAKEStartRequestResult),
    ]
