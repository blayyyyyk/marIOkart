from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
DWC_STATE_INIT = 0
DWC_STATE_AVAILABLE_CHECK = 1
DWC_STATE_LOGIN = 2
DWC_STATE_ONLINE = 3
DWC_STATE_UPDATE_SERVERS = 4
DWC_STATE_MATCHING = 5
DWC_STATE_CONNECTED = 6
DWC_STATE_NUM = 7

DWC_STATE_INIT = 0
DWC_STATE_AVAILABLE_CHECK = 1
DWC_STATE_LOGIN = 2
DWC_STATE_ONLINE = 3
DWC_STATE_UPDATE_SERVERS = 4
DWC_STATE_MATCHING = 5
DWC_STATE_CONNECTED = 6
DWC_STATE_NUM = 7


class DWCstControl(Structure):
    _fields_ = [
        ('gt2Socket', c_int),
        ('gt2Callbacks', c_int),
        ('gt2SendBufSize', c_int),
        ('gt2RecvBufSize', c_int),
        ('gpObj', c_int),
        ('userdata', POINTER32(c_int)),
        ('state', DWCState),
        ('lastState', DWCState),
        ('aid', c_int),
        ('ownCloseFlag', c_int),
        ('playerName', (c_int * 26)),
        ('profileID', c_int),
        ('gameName', POINTER32(c_char)),
        ('secretKey', POINTER32(c_char)),
        ('loginCallback', c_int),
        ('loginParam', c_void_p32),
        ('updateServersCallback', c_int),
        ('updateServersParam', c_void_p32),
        ('matchedCallback', c_int),
        ('matchedParam', c_void_p32),
        ('matchedSCCallback', c_int),
        ('matchedSCParam', c_void_p32),
        ('closedCallback', DWCConnectionClosedCallback),
        ('closedParam', c_void_p32),
        ('logcnt', c_int),
        ('friendcnt', c_int),
        ('matchcnt', c_int),
        ('transinfo', c_int),
    ]

class DWCstConnectionInfo(Structure):
    _fields_ = [
        ('index', c_int),
        ('aid', c_int),
        ('reserve', c_int),
        ('param', c_void_p32),
    ]
DWCState = c_int
DWCConnectionClosedCallback = c_void_p32
