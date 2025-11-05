from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
DWC_LOGIN_STATE_INIT = 0
DWC_LOGIN_STATE_REMOTE_AUTH = 1
DWC_LOGIN_STATE_CONNECTING = 2
DWC_LOGIN_STATE_GPGETINFO = 3
DWC_LOGIN_STATE_GPSETINFO = 4
DWC_LOGIN_STATE_CONNECTED = 5
DWC_LOGIN_STATE_NUM = 6

DWC_LOGIN_STATE_INIT = 0
DWC_LOGIN_STATE_REMOTE_AUTH = 1
DWC_LOGIN_STATE_CONNECTING = 2
DWC_LOGIN_STATE_GPGETINFO = 3
DWC_LOGIN_STATE_GPSETINFO = 4
DWC_LOGIN_STATE_CONNECTED = 5
DWC_LOGIN_STATE_NUM = 6


class DWCstLoginControl(Structure):
    _fields_ = [
        ('pGpObj', POINTER32(c_int)),
        ('state', DWCLoginState),
        ('productID', c_int),
        ('gamecode', c_int),
        ('playerName', POINTER32(c_int)),
        ('callback', DWCLoginCallback),
        ('param', c_void_p32),
        ('userdata', POINTER32(c_int)),
        ('bmwork', c_void_p32),
        ('http', c_void_p32),
        ('startTick', c_int),
        ('connectFlag', c_int),
        ('connectTick', c_int),
        ('tempLoginId', c_int),
        ('authToken', c_char),
        ('partnerChallenge', c_char),
        ('username', c_char),
        ('gpconnectresponsearg', c_int),
    ]
DWCLoginCallback = c_void_p32
DWCLoginState = c_int
