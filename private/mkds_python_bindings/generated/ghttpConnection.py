from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.ghttp import *
from private.mkds_python_bindings.generated.ghttpBuffer import *
from private.mkds_python_bindings.generated.ghttpEncryption import *
from private.mkds_python_bindings.generated.ghttpPost import *
from private.mkds_python_bindings.generated.gsPlatform import *
from private.mkds_python_bindings.generated.gsPlatformSocket import *
from private.mkds_python_bindings.generated.gsPlatformUtil import *

GHIGET = 0
GHISAVE = 1
GHISTREAM = 2
GHIHEAD = 3
GHIPOST = 4

GHIGET = 0
GHISAVE = 1
GHISTREAM = 2
GHIHEAD = 3
GHIPOST = 4

CRHeader = 0
CRChunk = 1
CRCRLF = 2
CRFooter = 3

CRHeader = 0
CRChunk = 1
CRCRLF = 2
CRFooter = 3

GHIHttp = 0
GHIHttps = 1

GHIHttp = 0
GHIHttps = 1

GHIGET = 0
GHISAVE = 1
GHISTREAM = 2
GHIHEAD = 3
GHIPOST = 4

GHIGET = 0
GHISAVE = 1
GHISTREAM = 2
GHIHEAD = 3
GHIPOST = 4

CRHeader = 0
CRChunk = 1
CRCRLF = 2
CRFooter = 3

CRHeader = 0
CRChunk = 1
CRCRLF = 2
CRFooter = 3

GHIHttp = 0
GHIHttps = 1

GHIHttp = 0
GHIHttps = 1

GHIGET = 0
GHISAVE = 1
GHISTREAM = 2
GHIHEAD = 3
GHIPOST = 4

GHIGET = 0
GHISAVE = 1
GHISTREAM = 2
GHIHEAD = 3
GHIPOST = 4

CRHeader = 0
CRChunk = 1
CRCRLF = 2
CRFooter = 3

CRHeader = 0
CRChunk = 1
CRCRLF = 2
CRFooter = 3

GHIHttp = 0
GHIHttps = 1

GHIHttp = 0
GHIHttps = 1

GHIGET = 0
GHISAVE = 1
GHISTREAM = 2
GHIHEAD = 3
GHIPOST = 4

GHIGET = 0
GHISAVE = 1
GHISTREAM = 2
GHIHEAD = 3
GHIPOST = 4

CRHeader = 0
CRChunk = 1
CRCRLF = 2
CRFooter = 3

CRHeader = 0
CRChunk = 1
CRCRLF = 2
CRFooter = 3

GHIHttp = 0
GHIHttps = 1

GHIHttp = 0
GHIHttps = 1


class GHIConnection(Structure):
    _fields_ = [
        ('inUse', GHTTPBool),
        ('request', GHTTPRequest),
        ('uniqueID', c_int),
        ('type', GHIRequestType),
        ('state', GHTTPState),
        ('URL', POINTER32(c_char)),
        ('serverAddress', POINTER32(c_char)),
        ('serverIP', c_int),
        ('serverPort', c_short),
        ('requestPath', POINTER32(c_char)),
        ('protocol', GHIProtocol),
        ('sendHeaders', POINTER32(c_char)),
        ('saveFile', POINTER32(FILE)),
        ('blocking', GHTTPBool),
        ('persistConnection', GHTTPBool),
        ('result', GHTTPResult),
        ('progressCallback', ghttpProgressCallback),
        ('completedCallback', ghttpCompletedCallback),
        ('callbackParam', c_void_p32),
        ('socket', SOCKET),
        ('socketError', c_int),
        ('sendBuffer', GHIBuffer),
        ('encodeBuffer', GHIBuffer),
        ('recvBuffer', GHIBuffer),
        ('decodeBuffer', GHIBuffer),
        ('getFileBuffer', GHIBuffer),
        ('userBufferSupplied', GHTTPBool),
        ('statusMajorVersion', c_int),
        ('statusMinorVersion', c_int),
        ('statusCode', c_int),
        ('statusStringIndex', c_int),
        ('headerStringIndex', c_int),
        ('completed', GHTTPBool),
        ('fileBytesReceived', GHTTPByteCount),
        ('totalSize', GHTTPByteCount),
        ('redirectURL', POINTER32(c_char)),
        ('redirectCount', c_int),
        ('chunkedTransfer', GHTTPBool),
        ('chunkHeader', (c_char * 11)),
        ('chunkHeaderLen', c_int),
        ('chunkBytesLeft', c_int),
        ('chunkReadingState', CRState),
        ('processing', GHTTPBool),
        ('connectionClosed', GHTTPBool),
        ('throttle', GHTTPBool),
        ('lastThrottleRecv', gsi_time),
        ('post', GHTTPPost),
        ('postingState', GHIPostingState),
        ('maxRecvTime', gsi_time),
        ('proxyOverrideServer', POINTER32(c_char)),
        ('proxyOverridePort', c_short),
        ('encryptor', GHIEncryptor),
        ('handle', GSIResolveHostnameHandle),
    ]
CRState = c_int
GHIProtocol = c_int
GHIRequestType = c_int
