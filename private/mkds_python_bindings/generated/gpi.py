from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.darray import *
from private.mkds_python_bindings.generated.gp import *
from private.mkds_python_bindings.generated.gpiBuffer import *
from private.mkds_python_bindings.generated.gpiCallback import *
from private.mkds_python_bindings.generated.gpiOperation import *
from private.mkds_python_bindings.generated.gpiPeer import *
from private.mkds_python_bindings.generated.gpiProfile import *
from private.mkds_python_bindings.generated.gsPlatform import *
from private.mkds_python_bindings.generated.gsPlatformSocket import *

GPIFalse = 0
GPITrue = 1

GPIFalse = 0
GPITrue = 1

GPIFalse = 0
GPITrue = 1

GPIFalse = 0
GPITrue = 1

GPIFalse = 0
GPITrue = 1

GPIFalse = 0
GPITrue = 1

GPIFalse = 0
GPITrue = 1

GPIFalse = 0
GPITrue = 1

GPIFalse = 0
GPITrue = 1

GPIFalse = 0
GPITrue = 1

GPIFalse = 0
GPITrue = 1

GPIFalse = 0
GPITrue = 1

GPIFalse = 0
GPITrue = 1

GPIFalse = 0
GPITrue = 1

GPIFalse = 0
GPITrue = 1

GPIFalse = 0
GPITrue = 1

GPIFalse = 0
GPITrue = 1

GPIFalse = 0
GPITrue = 1

GPIFalse = 0
GPITrue = 1

GPIFalse = 0
GPITrue = 1

GPIFalse = 0
GPITrue = 1

GPIFalse = 0
GPITrue = 1

GPIFalse = 0
GPITrue = 1

GPIFalse = 0
GPITrue = 1

GPIFalse = 0
GPITrue = 1

GPIFalse = 0
GPITrue = 1

GPIFalse = 0
GPITrue = 1

GPIFalse = 0
GPITrue = 1


class GPIConnection(Structure):
    _fields_ = [
        ('errorString', (c_char * 256)),
        ('infoCaching', GPIBool),
        ('infoCachingBuddyOnly', GPIBool),
        ('simulation', GPIBool),
        ('firewall', GPIBool),
        ('nick', (c_char * 31)),
        ('uniquenick', (c_char * 21)),
        ('email', (c_char * 51)),
        ('password', (c_char * 31)),
        ('sessKey', c_int),
        ('userid', c_int),
        ('profileid', c_int),
        ('partnerID', c_int),
        ('callbacks', (GPICallback * 9)),
        ('cmSocket', SOCKET),
        ('connectState', c_int),
        ('socketBuffer', GPIBuffer),
        ('inputBuffer', POINTER32(c_char)),
        ('inputBufferSize', c_int),
        ('outputBuffer', GPIBuffer),
        ('mHeader', (c_char * 16)),
        ('peerPort', c_short),
        ('nextOperationID', c_int),
        ('numSearches', c_int),
        ('lastStatusState', GPEnum),
        ('hostIp', c_int),
        ('hostPrivateIp', c_int),
        ('queryPort', c_short),
        ('hostPort', c_short),
        ('sessionFlags', c_int),
        ('richStatus', (c_char * 256)),
        ('gameType', (c_char * 33)),
        ('gameVariant', (c_char * 33)),
        ('gameMapName', (c_char * 33)),
        ('extendedInfoKeys', DArray),
        ('lastStatusString', (c_char * 256)),
        ('lastLocationString', (c_char * 256)),
        ('errorCode', GPErrorCode),
        ('fatalError', GPIBool),
        ('diskCache', POINTER32(FILE)),
        ('operationList', POINTER32(GPIOperation)),
        ('profileList', GPIProfileList),
        ('peerList', POINTER32(GPIPeer)),
        ('callbackList', POINTER32(GPICallbackData)),
        ('lastCallback', POINTER32(GPICallbackData)),
        ('updateproBuffer', GPIBuffer),
        ('updateuiBuffer', GPIBuffer),
        ('transfers', DArray),
        ('nextTransferID', c_int),
        ('productID', c_int),
        ('namespaceID', c_int),
        ('loginTicket', (c_char * 25)),
        ('quietModeFlags', GPEnum),
        ('kaTransmit', gsi_time),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_dwc_gs_gp_gpi_h_72_9(Structure):
    _fields_ = [
        ('errorString', (c_char * 256)),
        ('infoCaching', GPIBool),
        ('infoCachingBuddyOnly', GPIBool),
        ('simulation', GPIBool),
        ('firewall', GPIBool),
        ('nick', (c_char * 31)),
        ('uniquenick', (c_char * 21)),
        ('email', (c_char * 51)),
        ('password', (c_char * 31)),
        ('sessKey', c_int),
        ('userid', c_int),
        ('profileid', c_int),
        ('partnerID', c_int),
        ('callbacks', (GPICallback * 9)),
        ('cmSocket', SOCKET),
        ('connectState', c_int),
        ('socketBuffer', GPIBuffer),
        ('inputBuffer', POINTER32(c_char)),
        ('inputBufferSize', c_int),
        ('outputBuffer', GPIBuffer),
        ('mHeader', (c_char * 16)),
        ('peerPort', c_short),
        ('nextOperationID', c_int),
        ('numSearches', c_int),
        ('lastStatusState', GPEnum),
        ('hostIp', c_int),
        ('hostPrivateIp', c_int),
        ('queryPort', c_short),
        ('hostPort', c_short),
        ('sessionFlags', c_int),
        ('richStatus', (c_char * 256)),
        ('gameType', (c_char * 33)),
        ('gameVariant', (c_char * 33)),
        ('gameMapName', (c_char * 33)),
        ('extendedInfoKeys', DArray),
        ('lastStatusString', (c_char * 256)),
        ('lastLocationString', (c_char * 256)),
        ('errorCode', GPErrorCode),
        ('fatalError', GPIBool),
        ('diskCache', POINTER32(FILE)),
        ('operationList', POINTER32(GPIOperation)),
        ('profileList', GPIProfileList),
        ('peerList', POINTER32(c_int)),
        ('callbackList', POINTER32(GPICallbackData)),
        ('lastCallback', POINTER32(GPICallbackData)),
        ('updateproBuffer', GPIBuffer),
        ('updateuiBuffer', GPIBuffer),
        ('transfers', DArray),
        ('nextTransferID', c_int),
        ('productID', c_int),
        ('namespaceID', c_int),
        ('loginTicket', (c_char * 25)),
        ('quietModeFlags', GPEnum),
        ('kaTransmit', gsi_time),
    ]
GPIBool = c_int
