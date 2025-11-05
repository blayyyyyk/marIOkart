from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
DWC_TRANSPORT_SEND_READY = 0
DWC_TRANSPORT_SEND_BUSY = 1
DWC_TRANSPORT_SEND_LAST = 2

DWC_TRANSPORT_RECV_NOBUF = 0
DWC_TRANSPORT_RECV_HEADER = 1
DWC_TRANSPORT_RECV_BODY = 2
DWC_TRANSPORT_RECV_SYSTEM_DATA = 3
DWC_TRANSPORT_RECV_ERROR = 4
DWC_TRANSPORT_RECV_LAST = 5

DWC_SEND_TYPE_INVALID = 0
DWC_SEND_TYPE_USERDATA = 1
DWC_SEND_TYPE_MATCH_SYN = 2
DWC_SEND_TYPE_MATCH_SYN_ACK = 3
DWC_SEND_TYPE_MATCH_ACK = 4
DWC_SEND_TYPE_MAX = 5


class DWCstTransportConnection(Structure):
    _fields_ = [
        ('sendBuffer', POINTER32(c_int)),
        ('recvBuffer', POINTER32(c_int)),
        ('recvBufferSize', c_int),
        ('sendingSize', c_int),
        ('recvingSize', c_int),
        ('requestSendSize', c_int),
        ('requestRecvSize', c_int),
        ('sendState', c_int),
        ('recvState', c_int),
        ('lastRecvState', c_int),
        ('pads', (c_int * 3)),
        ('lastRecvType', c_int),
        ('previousRecvTick', c_int),
        ('recvTimeoutTime', c_int),
        ('sendDelay', c_int),
        ('recvDelay', c_int),
    ]

class DWCstTransportInfo(Structure):
    _fields_ = [
        ('connections', DWCTransportConnection),
        ('sendCallback', DWCUserSendCallback),
        ('recvCallback', DWCUserRecvCallback),
        ('recvTimeoutCallback', DWCUserRecvTimeoutCallback),
        ('pingCallback', DWCUserPingCallback),
        ('sendSplitMax', c_int),
        ('context', c_int),
        ('seed', c_int),
        ('delayedSend', c_int),
        ('delayedRecv', c_int),
        ('sendDrop', c_int),
        ('recvDrop', c_int),
        ('pads', (c_int * 2)),
    ]

class DWCstTransportHeader(Structure):
    _fields_ = [
        ('size', c_int),
        ('type', c_int),
        ('magicStrings', (c_char * 2)),
    ]

class DWCstDelayedMessage(Structure):
    _fields_ = [
        ('connection', c_int),
        ('filterID', c_int),
        ('message', POINTER32(c_int)),
        ('size', c_int),
        ('reliable', c_int),
        ('startTime', c_int),
        ('delayTime', c_int),
    ]
DWCUserRecvTimeoutCallback = c_void_p32
DWCUserRecvCallback = c_void_p32
DWCUserSendCallback = c_void_p32
DWCUserPingCallback = c_void_p32
