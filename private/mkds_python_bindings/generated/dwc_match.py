from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
DWC_MATCH_OPTION_MIN_COMPLETE = 0
DWC_MATCH_OPTION_SC_CONNECT_BLOCK = 1
DWC_MATCH_OPTION_NUM = 2

DWC_MATCH_OPTION_MIN_COMPLETE = 0
DWC_MATCH_OPTION_SC_CONNECT_BLOCK = 1
DWC_MATCH_OPTION_NUM = 2

DWC_SET_MATCH_OPT_RESULT_SUCCESS = 0
DWC_SET_MATCH_OPT_RESULT_E_BAD_STATE = 1
DWC_SET_MATCH_OPT_RESULT_E_INVALID = 2
DWC_SET_MATCH_OPT_RESULT_E_PARAM = 3
DWC_SET_MATCH_OPT_RESULT_E_ALLOC = 4
DWC_SET_MATCH_OPT_RESULT_NUM = 5

DWC_MATCH_STATE_INIT = 0
DWC_MATCH_STATE_CL_WAITING = 1
DWC_MATCH_STATE_CL_SEARCH_OWN = 2
DWC_MATCH_STATE_CL_SEARCH_HOST = 3
DWC_MATCH_STATE_CL_WAIT_RESV = 4
DWC_MATCH_STATE_CL_SEARCH_NN_HOST = 5
DWC_MATCH_STATE_CL_NN = 6
DWC_MATCH_STATE_CL_GT2 = 7
DWC_MATCH_STATE_CL_CANCEL_SYN = 8
DWC_MATCH_STATE_CL_SYN = 9
DWC_MATCH_STATE_SV_WAITING = 10
DWC_MATCH_STATE_SV_OWN_NN = 11
DWC_MATCH_STATE_SV_OWN_GT2 = 12
DWC_MATCH_STATE_SV_WAIT_CL_LINK = 13
DWC_MATCH_STATE_SV_CANCEL_SYN = 14
DWC_MATCH_STATE_SV_CANCEL_SYN_WAIT = 15
DWC_MATCH_STATE_SV_SYN = 16
DWC_MATCH_STATE_SV_SYN_WAIT = 17
DWC_MATCH_STATE_WAIT_CLOSE = 18
DWC_MATCH_STATE_SV_POLL_TIMEOUT = 19
DWC_MATCH_STATE_NUM = 20

DWC_MATCH_STATE_INIT = 0
DWC_MATCH_STATE_CL_WAITING = 1
DWC_MATCH_STATE_CL_SEARCH_OWN = 2
DWC_MATCH_STATE_CL_SEARCH_HOST = 3
DWC_MATCH_STATE_CL_WAIT_RESV = 4
DWC_MATCH_STATE_CL_SEARCH_NN_HOST = 5
DWC_MATCH_STATE_CL_NN = 6
DWC_MATCH_STATE_CL_GT2 = 7
DWC_MATCH_STATE_CL_CANCEL_SYN = 8
DWC_MATCH_STATE_CL_SYN = 9
DWC_MATCH_STATE_SV_WAITING = 10
DWC_MATCH_STATE_SV_OWN_NN = 11
DWC_MATCH_STATE_SV_OWN_GT2 = 12
DWC_MATCH_STATE_SV_WAIT_CL_LINK = 13
DWC_MATCH_STATE_SV_CANCEL_SYN = 14
DWC_MATCH_STATE_SV_CANCEL_SYN_WAIT = 15
DWC_MATCH_STATE_SV_SYN = 16
DWC_MATCH_STATE_SV_SYN_WAIT = 17
DWC_MATCH_STATE_WAIT_CLOSE = 18
DWC_MATCH_STATE_SV_POLL_TIMEOUT = 19
DWC_MATCH_STATE_NUM = 20

DWC_MATCH_TYPE_ANYBODY = 0
DWC_MATCH_TYPE_FRIEND = 1
DWC_MATCH_TYPE_SC_SV = 2
DWC_MATCH_TYPE_SC_CL = 3
DWC_MATCH_TYPE_NUM = 4


class DWCstMatchOptMinComplete(Structure):
    _fields_ = [
        ('valid', c_int),
        ('minEntry', c_int),
        ('pad', (c_int * 2)),
        ('timeout', c_int),
    ]

class DWCstNNInfo(Structure):
    _fields_ = [
        ('isQR2', c_int),
        ('retryCount', c_int),
        ('port', c_int),
        ('ip', c_int),
        ('cookie', c_int),
    ]

class DWCstMatchCommandControl(Structure):
    _fields_ = [
        ('command', c_int),
        ('count', c_int),
        ('port', c_int),
        ('ip', c_int),
        ('data', (c_int * 32)),
        ('profileID', c_int),
        ('len', c_int),
        ('sendTime', c_int),
    ]

class DWCstMatchControl(Structure):
    _fields_ = [
        ('pGpObj', POINTER32(c_int)),
        ('pGt2Socket', POINTER32(c_int)),
        ('gt2Callbacks', POINTER32(c_int)),
        ('gt2ConnectCount', c_int),
        ('gt2NumConnection', c_int),
        ('gt2NumValidConn', c_int),
        ('pad1', c_int),
        ('qr2Obj', c_int),
        ('qr2NNFinishCount', c_int),
        ('qr2MatchType', c_int),
        ('qr2NumEntry', c_int),
        ('qr2IsReserved', c_int),
        ('qr2ShutdownFlag', c_int),
        ('pad2', c_int),
        ('qr2Port', c_int),
        ('qr2IP', c_int),
        ('qr2Reservation', c_int),
        ('qr2IPList', (c_int * 32)),
        ('qr2PortList', (c_int * 32)),
        ('sbObj', c_int),
        ('sbUpdateFlag', c_int),
        ('sbUpdateTick', c_int),
        ('sbPidList', (c_int * 32)),
        ('sbUpdateRequestTick', c_int),
        ('nnRecvCount', c_int),
        ('nnFailureCount', c_int),
        ('nnCookieRand', c_int),
        ('nnLastCookie', c_int),
        ('nnFailedTime', c_int),
        ('nnFinishTime', c_int),
        ('nnInfo', DWCNNInfo),
        ('state', DWCMatchState),
        ('clLinkProgress', c_int),
        ('friendCount', c_int),
        ('distantFriend', c_int),
        ('resvWaitCount', c_int),
        ('closeState', c_int),
        ('cancelState', c_int),
        ('scResvRetryCount', c_int),
        ('synResendCount', c_int),
        ('cancelSynResendCount', c_int),
        ('clWaitTimeoutCount', c_int),
        ('stopSCFlag', c_int),
        ('pad3', c_int),
        ('baseLatency', c_int),
        ('cancelBaseLatency', c_int),
        ('searchPort', c_int),
        ('pad4', c_int),
        ('searchIP', c_int),
        ('cmdResendFlag', c_int),
        ('cmdResendTick', c_int),
        ('cmdTimeoutTime', c_int),
        ('cmdTimeoutStartTick', c_int),
        ('synAckBit', c_int),
        ('cancelSynAckBit', c_int),
        ('friendAcceptBit', c_int),
        ('lastSynSent', c_int),
        ('lastCancelSynSent', c_int),
        ('closedTime', c_int),
        ('clWaitTime', c_int),
        ('profileID', c_int),
        ('reqProfileID', c_int),
        ('priorProfileID', c_int),
        ('cbEventPid', c_int),
        ('ipList', (c_int * 32)),
        ('portList', (c_int * 32)),
        ('aidList', (c_int * 32)),
        ('validAidBitmap', c_int),
        ('gameName', POINTER32(c_char)),
        ('secretKey', POINTER32(c_char)),
        ('friendList', POINTER32(c_int)),
        ('friendListLen', c_int),
        ('friendIdxList', (c_int * 64)),
        ('friendIdxListLen', c_int),
        ('svDataBak', (c_int * 33)),
        ('cmdCnt', DWCMatchCommandControl),
        ('matchedCallback', DWCMatchedSCCallback),
        ('matchedParam', c_void_p32),
        ('newClientCallback', DWCNewClientCallback),
        ('newClientParam', c_void_p32),
        ('evalCallback', DWCEvalPlayerCallback),
        ('evalParam', c_void_p32),
        ('stopSCCallback', DWCStopSCCallback),
        ('stopSCParam', c_void_p32),
    ]

class DWCstSBMessageHeader(Structure):
    _fields_ = [
        ('identifier', (c_char * 4)),
        ('version', c_int),
        ('command', c_int),
        ('size', c_int),
        ('qr2Port', c_int),
        ('qr2IP', c_int),
        ('profileID', c_int),
    ]

class DWCstSBMessage(Structure):
    _fields_ = [
        ('header', DWCSBMessageHeader),
        ('data', (c_int * 32)),
    ]

class DWCstGameMatchKeyData(Structure):
    _fields_ = [
        ('keyID', c_int),
        ('isStr', c_int),
        ('pad', c_int),
        ('keyStr', POINTER32(c_char)),
        ('value', c_void_p32),
    ]

class DWCstMatchOptMinCompleteIn(Structure):
    _fields_ = [
        ('valid', c_int),
        ('minEntry', c_int),
        ('retry', c_int),
        ('pad', c_int),
        ('timeout', c_int),
        ('recvBit', c_int),
        ('timeoutBit', c_int),
        ('startTime', c_int),
        ('lastPollTime', c_int),
    ]

class DWCstMatchOptSCBlock(Structure):
    _fields_ = [
        ('valid', c_int),
        ('lock', c_int),
        ('pad', c_int),
    ]
DWCMatchOptType = c_int
DWCMatchedCallback = c_void_p32
DWCNewClientCallback = c_void_p32
DWCMatchState = c_int
DWCMatchedSCCallback = c_void_p32
DWCEvalPlayerCallback = c_void_p32
DWCStopSCCallback = c_void_p32
