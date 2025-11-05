from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.chat import *
from private.mkds_python_bindings.generated.chatCrypt import *
from private.mkds_python_bindings.generated.gsPlatformSocket import *

ciNotConnected = 0
ciConnected = 1
ciDisconnected = 2

ciNotConnected = 0
ciConnected = 1
ciDisconnected = 2

ciNotConnected = 0
ciConnected = 1
ciDisconnected = 2

ciNotConnected = 0
ciConnected = 1
ciDisconnected = 2

ciNotConnected = 0
ciConnected = 1
ciDisconnected = 2

ciNotConnected = 0
ciConnected = 1
ciDisconnected = 2

ciNotConnected = 0
ciConnected = 1
ciDisconnected = 2

ciNotConnected = 0
ciConnected = 1
ciDisconnected = 2


class ciSocket(Structure):
    _fields_ = [
        ('sock', SOCKET),
        ('connectState', ciConnectState),
        ('serverAddress', (c_char * 256)),
        ('inputQueue', ciBuffer),
        ('outputQueue', ciBuffer),
        ('secure', CHATBool),
        ('inKey', gs_crypt_key),
        ('outKey', gs_crypt_key),
        ('lastMessage', ciServerMessage),
    ]

class ciBuffer(Structure):
    _fields_ = [
        ('buffer', POINTER32(c_char)),
        ('length', c_int),
        ('size', c_int),
    ]

class ciServerMessage(Structure):
    _fields_ = [
        ('message', POINTER32(c_char)),
        ('server', POINTER32(c_char)),
        ('nick', POINTER32(c_char)),
        ('user', POINTER32(c_char)),
        ('host', POINTER32(c_char)),
        ('command', POINTER32(c_char)),
        ('middle', POINTER32(c_char)),
        ('param', POINTER32(c_char)),
        ('params', POINTER32(POINTER32(c_char))),
        ('numParams', c_int),
    ]
ciConnectState = c_int
