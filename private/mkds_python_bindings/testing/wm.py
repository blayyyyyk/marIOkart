from ctypes import *
from private.mkds_python_bindings.testing.thread import *
from private.mkds_python_bindings.testing.tick import *
from private.mkds_python_bindings.testing.types import *

WM_APIID_INITIALIZE = 0
WM_APIID_RESET = 1
WM_APIID_END = 2
WM_APIID_ENABLE = 3
WM_APIID_DISABLE = 4
WM_APIID_POWER_ON = 5
WM_APIID_POWER_OFF = 6
WM_APIID_SET_P_PARAM = 7
WM_APIID_START_PARENT = 8
WM_APIID_END_PARENT = 9
WM_APIID_START_SCAN = 10
WM_APIID_END_SCAN = 11
WM_APIID_START_CONNECT = 12
WM_APIID_DISCONNECT = 13
WM_APIID_START_MP = 14
WM_APIID_SET_MP_DATA = 15
WM_APIID_END_MP = 16
WM_APIID_START_DCF = 17
WM_APIID_SET_DCF_DATA = 18
WM_APIID_END_DCF = 19
WM_APIID_SET_WEPKEY = 20
WM_APIID_START_KS = 21
WM_APIID_END_KS = 22
WM_APIID_GET_KEYSET = 23
WM_APIID_SET_GAMEINFO = 24
WM_APIID_SET_BEACON_IND = 25
WM_APIID_START_TESTMODE = 26
WM_APIID_STOP_TESTMODE = 27
WM_APIID_VALARM_MP = 28
WM_APIID_SET_LIFETIME = 29
WM_APIID_MEASURE_CHANNEL = 30
WM_APIID_INIT_W_COUNTER = 31
WM_APIID_GET_W_COUNTER = 32
WM_APIID_SET_ENTRY = 33
WM_APIID_AUTO_DEAUTH = 34
WM_APIID_SET_MP_PARAMETER = 35
WM_APIID_SET_BEACON_PERIOD = 36
WM_APIID_AUTO_DISCONNECT = 37
WM_APIID_START_SCAN_EX = 38
WM_APIID_SET_WEPKEY_EX = 39
WM_APIID_SET_PS_MODE = 40
WM_APIID_START_TESTRXMODE = 41
WM_APIID_STOP_TESTRXMODE = 42
WM_APIID_KICK_MP_PARENT = 43
WM_APIID_KICK_MP_CHILD = 44
WM_APIID_KICK_MP_RESUME = 45
WM_APIID_ASYNC_KIND_MAX = 46
WM_APIID_INDICATION = 128
WM_APIID_PORT_SEND = 129
WM_APIID_PORT_RECV = 130
WM_APIID_READ_STATUS = 131
WM_APIID_UNKNOWN = 255

WM_APIID_INITIALIZE = 0
WM_APIID_RESET = 1
WM_APIID_END = 2
WM_APIID_ENABLE = 3
WM_APIID_DISABLE = 4
WM_APIID_POWER_ON = 5
WM_APIID_POWER_OFF = 6
WM_APIID_SET_P_PARAM = 7
WM_APIID_START_PARENT = 8
WM_APIID_END_PARENT = 9
WM_APIID_START_SCAN = 10
WM_APIID_END_SCAN = 11
WM_APIID_START_CONNECT = 12
WM_APIID_DISCONNECT = 13
WM_APIID_START_MP = 14
WM_APIID_SET_MP_DATA = 15
WM_APIID_END_MP = 16
WM_APIID_START_DCF = 17
WM_APIID_SET_DCF_DATA = 18
WM_APIID_END_DCF = 19
WM_APIID_SET_WEPKEY = 20
WM_APIID_START_KS = 21
WM_APIID_END_KS = 22
WM_APIID_GET_KEYSET = 23
WM_APIID_SET_GAMEINFO = 24
WM_APIID_SET_BEACON_IND = 25
WM_APIID_START_TESTMODE = 26
WM_APIID_STOP_TESTMODE = 27
WM_APIID_VALARM_MP = 28
WM_APIID_SET_LIFETIME = 29
WM_APIID_MEASURE_CHANNEL = 30
WM_APIID_INIT_W_COUNTER = 31
WM_APIID_GET_W_COUNTER = 32
WM_APIID_SET_ENTRY = 33
WM_APIID_AUTO_DEAUTH = 34
WM_APIID_SET_MP_PARAMETER = 35
WM_APIID_SET_BEACON_PERIOD = 36
WM_APIID_AUTO_DISCONNECT = 37
WM_APIID_START_SCAN_EX = 38
WM_APIID_SET_WEPKEY_EX = 39
WM_APIID_SET_PS_MODE = 40
WM_APIID_START_TESTRXMODE = 41
WM_APIID_STOP_TESTRXMODE = 42
WM_APIID_KICK_MP_PARENT = 43
WM_APIID_KICK_MP_CHILD = 44
WM_APIID_KICK_MP_RESUME = 45
WM_APIID_ASYNC_KIND_MAX = 46
WM_APIID_INDICATION = 128
WM_APIID_PORT_SEND = 129
WM_APIID_PORT_RECV = 130
WM_APIID_READ_STATUS = 131
WM_APIID_UNKNOWN = 255

WM_PORT_RAWDATA = 0
WM_PORT_BT = 1
WM_PORT_RESERVE_02 = 2
WM_PORT_RESERVE_03 = 3
WM_PORT_RESERVE_10 = 8
WM_PORT_RESERVE_11 = 9
WM_PORT_RESERVE_12 = 10
WM_PORT_RESERVE_13 = 11

WM_PORT_RAWDATA = 0
WM_PORT_BT = 1
WM_PORT_RESERVE_02 = 2
WM_PORT_RESERVE_03 = 3
WM_PORT_RESERVE_10 = 8
WM_PORT_RESERVE_11 = 9
WM_PORT_RESERVE_12 = 10
WM_PORT_RESERVE_13 = 11

WM_PRIORITY_URGENT = 0
WM_PRIORITY_HIGH = 1
WM_PRIORITY_NORMAL = 2
WM_PRIORITY_LOW = 3

WM_PRIORITY_URGENT = 0
WM_PRIORITY_HIGH = 1
WM_PRIORITY_NORMAL = 2
WM_PRIORITY_LOW = 3

WM_ERRCODE_SUCCESS = 0
WM_ERRCODE_FAILED = 1
WM_ERRCODE_OPERATING = 2
WM_ERRCODE_ILLEGAL_STATE = 3
WM_ERRCODE_WM_DISABLE = 4
WM_ERRCODE_NO_KEYSET = 5
WM_ERRCODE_NO_DATASET = 5
WM_ERRCODE_INVALID_PARAM = 6
WM_ERRCODE_NO_CHILD = 7
WM_ERRCODE_FIFO_ERROR = 8
WM_ERRCODE_TIMEOUT = 9
WM_ERRCODE_SEND_QUEUE_FULL = 10
WM_ERRCODE_NO_ENTRY = 11
WM_ERRCODE_OVER_MAX_ENTRY = 12
WM_ERRCODE_INVALID_POLLBITMAP = 13
WM_ERRCODE_NO_DATA = 14
WM_ERRCODE_SEND_FAILED = 15
WM_ERRCODE_DCF_TEST = 16
WM_ERRCODE_WL_INVALID_PARAM = 17
WM_ERRCODE_WL_LENGTH_ERR = 18
WM_ERRCODE_FLASH_ERROR = 19
WM_ERRCODE_MAX = 20

WM_ERRCODE_SUCCESS = 0
WM_ERRCODE_FAILED = 1
WM_ERRCODE_OPERATING = 2
WM_ERRCODE_ILLEGAL_STATE = 3
WM_ERRCODE_WM_DISABLE = 4
WM_ERRCODE_NO_KEYSET = 5
WM_ERRCODE_NO_DATASET = 5
WM_ERRCODE_INVALID_PARAM = 6
WM_ERRCODE_NO_CHILD = 7
WM_ERRCODE_FIFO_ERROR = 8
WM_ERRCODE_TIMEOUT = 9
WM_ERRCODE_SEND_QUEUE_FULL = 10
WM_ERRCODE_NO_ENTRY = 11
WM_ERRCODE_OVER_MAX_ENTRY = 12
WM_ERRCODE_INVALID_POLLBITMAP = 13
WM_ERRCODE_NO_DATA = 14
WM_ERRCODE_SEND_FAILED = 15
WM_ERRCODE_DCF_TEST = 16
WM_ERRCODE_WL_INVALID_PARAM = 17
WM_ERRCODE_WL_LENGTH_ERR = 18
WM_ERRCODE_FLASH_ERROR = 19
WM_ERRCODE_MAX = 20

WM_STATECODE_PARENT_START = 0
WM_STATECODE_BEACON_SENT = 2
WM_STATECODE_SCAN_START = 3
WM_STATECODE_PARENT_NOT_FOUND = 4
WM_STATECODE_PARENT_FOUND = 5
WM_STATECODE_CONNECT_START = 6
WM_STATECODE_BEACON_LOST = 8
WM_STATECODE_CONNECTED = 7
WM_STATECODE_CHILD_CONNECTED = 7
WM_STATECODE_DISCONNECTED = 9
WM_STATECODE_DISCONNECTED_FROM_MYSELF = 26
WM_STATECODE_MP_START = 10
WM_STATECODE_MPEND_IND = 11
WM_STATECODE_MP_IND = 12
WM_STATECODE_MPACK_IND = 13
WM_STATECODE_DCF_START = 14
WM_STATECODE_DCF_IND = 15
WM_STATECODE_BEACON_RECV = 16
WM_STATECODE_DISASSOCIATE = 17
WM_STATECODE_REASSOCIATE = 18
WM_STATECODE_AUTHENTICATE = 19
WM_STATECODE_PORT_INIT = 25
WM_STATECODE_PORT_SEND = 20
WM_STATECODE_PORT_RECV = 21
WM_STATECODE_FIFO_ERROR = 22
WM_STATECODE_INFORMATION = 23
WM_STATECODE_UNKNOWN = 24
WM_STATECODE_MAX = 27

WM_STATECODE_PARENT_START = 0
WM_STATECODE_BEACON_SENT = 2
WM_STATECODE_SCAN_START = 3
WM_STATECODE_PARENT_NOT_FOUND = 4
WM_STATECODE_PARENT_FOUND = 5
WM_STATECODE_CONNECT_START = 6
WM_STATECODE_BEACON_LOST = 8
WM_STATECODE_CONNECTED = 7
WM_STATECODE_CHILD_CONNECTED = 7
WM_STATECODE_DISCONNECTED = 9
WM_STATECODE_DISCONNECTED_FROM_MYSELF = 26
WM_STATECODE_MP_START = 10
WM_STATECODE_MPEND_IND = 11
WM_STATECODE_MP_IND = 12
WM_STATECODE_MPACK_IND = 13
WM_STATECODE_DCF_START = 14
WM_STATECODE_DCF_IND = 15
WM_STATECODE_BEACON_RECV = 16
WM_STATECODE_DISASSOCIATE = 17
WM_STATECODE_REASSOCIATE = 18
WM_STATECODE_AUTHENTICATE = 19
WM_STATECODE_PORT_INIT = 25
WM_STATECODE_PORT_SEND = 20
WM_STATECODE_PORT_RECV = 21
WM_STATECODE_FIFO_ERROR = 22
WM_STATECODE_INFORMATION = 23
WM_STATECODE_UNKNOWN = 24
WM_STATECODE_MAX = 27

WM_STATE_READY = 0
WM_STATE_STOP = 1
WM_STATE_IDLE = 2
WM_STATE_CLASS1 = 3
WM_STATE_TESTMODE = 4
WM_STATE_SCAN = 5
WM_STATE_CONNECT = 6
WM_STATE_PARENT = 7
WM_STATE_CHILD = 8
WM_STATE_MP_PARENT = 9
WM_STATE_MP_CHILD = 10
WM_STATE_DCF_CHILD = 11
WM_STATE_TESTMODE_RX = 12
WM_STATE_MAX = 13

WM_STATE_READY = 0
WM_STATE_STOP = 1
WM_STATE_IDLE = 2
WM_STATE_CLASS1 = 3
WM_STATE_TESTMODE = 4
WM_STATE_SCAN = 5
WM_STATE_CONNECT = 6
WM_STATE_PARENT = 7
WM_STATE_CHILD = 8
WM_STATE_MP_PARENT = 9
WM_STATE_MP_CHILD = 10
WM_STATE_DCF_CHILD = 11
WM_STATE_TESTMODE_RX = 12
WM_STATE_MAX = 13

WM_LINK_LEVEL_0 = 0
WM_LINK_LEVEL_1 = 1
WM_LINK_LEVEL_2 = 2
WM_LINK_LEVEL_3 = 3
WM_LINK_LEVEL_MAX = 4

WM_LINK_LEVEL_0 = 0
WM_LINK_LEVEL_1 = 1
WM_LINK_LEVEL_2 = 2
WM_LINK_LEVEL_3 = 3
WM_LINK_LEVEL_MAX = 4

WM_DS_STATE_READY = 0
WM_DS_STATE_START = 1
WM_DS_STATE_PAUSING = 2
WM_DS_STATE_PAUSED = 3
WM_DS_STATE_RETRY_SEND = 4
WM_DS_STATE_ERROR = 5

WM_DS_STATE_READY = 0
WM_DS_STATE_START = 1
WM_DS_STATE_PAUSING = 2
WM_DS_STATE_PAUSED = 3
WM_DS_STATE_RETRY_SEND = 4
WM_DS_STATE_ERROR = 5

WM_DISCONNECT_REASON_RESERVED = 0
WM_DISCONNECT_REASON_UNSPECIFIED = 1
WM_DISCONNECT_REASON_PREV_AUTH_INVALID = 2
WM_DISCONNECT_REASON_DEAUTH_LEAVING = 3
WM_DISCONNECT_REASON_INACTIVE = 4
WM_DISCONNECT_REASON_UNABLE_HANDLE = 5
WM_DISCONNECT_REASON_RX_CLASS2_FROM_NONAUTH_STA = 6
WM_DISCONNECT_REASON_RX_CLASS3_FROM_NONASSOC_STA = 7
WM_DISCONNECT_REASON_DISASSOC_LEAVING = 8
WM_DISCONNECT_REASON_ASSOC_STA_NOTAUTHED = 9
WM_DISCONNECT_REASON_NO_ENTRY = 19
WM_DISCONNECT_REASON_MP_LIFETIME = 32769
WM_DISCONNECT_REASON_TGID_CHANGED = 32770
WM_DISCONNECT_REASON_FATAL_ERROR = 32771
WM_DISCONNECT_REASON_FROM_MYSELF = 61441

WM_DISCONNECT_REASON_RESERVED = 0
WM_DISCONNECT_REASON_UNSPECIFIED = 1
WM_DISCONNECT_REASON_PREV_AUTH_INVALID = 2
WM_DISCONNECT_REASON_DEAUTH_LEAVING = 3
WM_DISCONNECT_REASON_INACTIVE = 4
WM_DISCONNECT_REASON_UNABLE_HANDLE = 5
WM_DISCONNECT_REASON_RX_CLASS2_FROM_NONAUTH_STA = 6
WM_DISCONNECT_REASON_RX_CLASS3_FROM_NONASSOC_STA = 7
WM_DISCONNECT_REASON_DISASSOC_LEAVING = 8
WM_DISCONNECT_REASON_ASSOC_STA_NOTAUTHED = 9
WM_DISCONNECT_REASON_NO_ENTRY = 19
WM_DISCONNECT_REASON_MP_LIFETIME = 32769
WM_DISCONNECT_REASON_TGID_CHANGED = 32770
WM_DISCONNECT_REASON_FATAL_ERROR = 32771
WM_DISCONNECT_REASON_FROM_MYSELF = 61441

WM_INFOCODE_NONE = 0
WM_INFOCODE_FATAL_ERROR = 1

WM_INFOCODE_NONE = 0
WM_INFOCODE_FATAL_ERROR = 1

WMApiid = c_int
WMCallbackFunc = u32
WMDataSharingState = c_int
WMDisconnectReason = c_int
WMErrCode = c_int
WMInfoCode = c_int
WMKeySetBuf = WMDataSharingInfo
WMLinkLevel = c_int
WMPort = c_int
WMPriorityLevel = c_int
WMState = c_int
WMStateCode = c_int
WMcallbackFunc = u32
WMkeySetBuf = WMDataSharingInfo


class WMDataSet(Structure):
    _fields_ = [
        ('aidBitmap', u16),
        ('receivedBitmap', u16),
        ('data', (u16 * 254)),
    ]

class WMDataSharingInfo(Structure):
    _fields_ = [
        ('ds', (WMDataSet * 4)),
        ('seqNum', (u16 * 4)),
        ('writeIndex', u16),
        ('sendIndex', u16),
        ('readIndex', u16),
        ('aidBitmap', u16),
        ('dataLength', u16),
        ('stationNumber', u16),
        ('dataSetLength', u16),
        ('port', u16),
        ('doubleMode', u16),
        ('currentSeqNum', u16),
        ('state', u16),
        ('reserved', (u16 * 1)),
    ]

class WMKeySet(Structure):
    _fields_ = [
        ('seqNum', u16),
        ('rsv', u16),
        ('key', (u16 * 16)),
    ]

class WMPortSendQueueData(Structure):
    _fields_ = [
        ('next', u16),
        ('port', u16),
        ('destBitmap', u16),
        ('restBitmap', u16),
        ('sentBitmap', u16),
        ('sendingBitmap', u16),
        ('padding', u16),
        ('size', u16),
        ('seqNo', u16),
        ('retryCount', u16),
        ('data', u32), #POINTER(u16)),
        ('callback', WMCallbackFunc),
        ('arg', u32),
    ]

class WMPortSendQueue(Structure):
    _fields_ = [
        ('head', u16),
        ('tail', u16),
    ]

class WMMpRecvBuf(Structure):
    _fields_ = [
        ('rsv1', (u16 * 3)),
        ('length', u16),
        ('rsv2', (u16 * 1)),
        ('ackTimeStamp', u16),
        ('timeStamp', u16),
        ('rate_rssi', u16),
        ('rsv3', (u16 * 2)),
        ('rsv4', (u16 * 2)),
        ('destAdrs', (u8 * 6)),
        ('srcAdrs', (u8 * 6)),
        ('rsv5', (u16 * 3)),
        ('seqCtrl', u16),
        ('txop', u16),
        ('bitmap', u16),
        ('wmHeader', u16),
        ('data', (u16 * 2)),
    ]

class WMMpRecvData(Structure):
    _fields_ = [
        ('length', u16),
        ('rate_rssi', u16),
        ('aid', u16),
        ('noResponse', u16),
        ('wmHeader', u16),
        ('cdata', (u16 * 1)),
    ]

class WMMpRecvHeader(Structure):
    _fields_ = [
        ('bitmap', u16),
        ('errBitmap', u16),
        ('count', u16),
        ('length', u16),
        ('txCount', u16),
        ('data', (WMMpRecvData * 1)),
    ]

class WMDcfRecvBuf(Structure):
    _fields_ = [
        ('frameID', u16),
        ('rsv1', (u16 * 2)),
        ('length', u16),
        ('rsv2', (u16 * 3)),
        ('rate_rssi', u16),
        ('rsv3', (u16 * 4)),
        ('destAdrs', (u8 * 6)),
        ('srcAdrs', (u8 * 6)),
        ('rsv4', (u16 * 4)),
        ('data', (u16 * 2)),
    ]

class WMParentParam(Structure):
    _fields_ = [
        ('userGameInfo', u32), #POINTER(u16)),
        ('userGameInfoLength', u16),
        ('padding', u16),
        ('ggid', u32),
        ('tgid', u16),
        ('entryFlag', u16),
        ('maxEntry', u16),
        ('multiBootFlag', u16),
        ('KS_Flag', u16),
        ('CS_Flag', u16),
        ('beaconPeriod', u16),
        ('rsv1', (u16 * 4)),
        ('rsv2', (u16 * 8)),
        ('channel', u16),
        ('parentMaxSize', u16),
        ('childMaxSize', u16),
        ('rsv', (u16 * 4)),
    ]

class WMGameInfo(Structure):
    _fields_ = [
        ('version', u16),
        ('padd0', u16),
        ('ggid', u32),
        ('tgid', u16),
        ('userGameInfoLength', u8),
        ('gameNameCount_attribute', u8),
        ('parentMaxSize', u16),
        ('childMaxSize', u16),
    ]

class union__anonymous_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_wm_common_wm_h_1314_5_(Union):
    _fields_ = [
        ('userGameInfo', (u16 * 56)),
        ('old_type', struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_wm_common_wm_h_1317_9_),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_wm_common_wm_h_1317_9_(Structure):
    _fields_ = [
        ('userName', (u16 * 4)),
        ('gameName', (u16 * 8)),
        ('padd1', (u16 * 44)),
    ]

class WMBssDesc(Structure):
    _fields_ = [
        ('length', u16),
        ('rssi', u16),
        ('bssid', (u8 * 6)),
        ('ssidLength', u16),
        ('ssid', (u8 * 32)),
        ('capaInfo', u16),
        ('rateSet', struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_wm_common_wm_h_1367_5_),
        ('beaconPeriod', u16),
        ('dtimPeriod', u16),
        ('channel', u16),
        ('cfpPeriod', u16),
        ('cfpMaxDuration', u16),
        ('gameInfoLength', u16),
        ('otherElementCount', u16),
        ('gameInfo', WMGameInfo),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_wm_common_wm_h_1367_5_(Structure):
    _fields_ = [
        ('basic', u16),
        ('support', u16),
    ]

class WMOtherElements(Structure):
    _fields_ = [
        ('count', u8),
        ('rsv', (u8 * 3)),
        ('element', (struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_wm_common_wm_h_1392_5_ * 16)),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_wm_common_wm_h_1392_5_(Structure):
    _fields_ = [
        ('id', u8),
        ('length', u8),
        ('rsv', (u8 * 2)),
        ('body', u32), #POINTER(u8)),
    ]

class WMScanParam(Structure):
    _fields_ = [
        ('scanBuf', u32), #POINTER(WMBssDesc)),
        ('channel', u16),
        ('maxChannelTime', u16),
        ('bssid', (u8 * 6)),
        ('rsv', (u16 * 9)),
    ]

class WMScanExParam(Structure):
    _fields_ = [
        ('scanBuf', u32), #POINTER(WMBssDesc)),
        ('scanBufSize', u16),
        ('channelList', u16),
        ('maxChannelTime', u16),
        ('bssid', (u8 * 6)),
        ('scanType', u16),
        ('ssidLength', u16),
        ('ssid', (u8 * 32)),
        ('ssidMatchLength', u16),
        ('rsv2', (u16 * 7)),
    ]

class WMMPParam(Structure):
    _fields_ = [
        ('mask', u32),
        ('minFrequency', u16),
        ('frequency', u16),
        ('maxFrequency', u16),
        ('parentSize', u16),
        ('childSize', u16),
        ('parentInterval', u16),
        ('childInterval', u16),
        ('parentVCount', u16),
        ('childVCount', u16),
        ('defaultRetryCount', u16),
        ('minPollBmpMode', u8),
        ('singlePacketMode', u8),
        ('ignoreFatalErrorMode', u8),
        ('ignoreSizePrecheckMode', u8),
    ]

class WMStatus(Structure):
    _fields_ = [
        ('state', u16),
        ('BusyApiid', u16),
        ('apiBusy', BOOL),
        ('scan_continue', BOOL),
        ('mp_flag', BOOL),
        ('dcf_flag', BOOL),
        ('ks_flag', BOOL),
        ('dcf_sendFlag', BOOL),
        ('VSyncFlag', BOOL),
        ('wlVersion', (u8 * 8)),
        ('macVersion', u16),
        ('rfVersion', u16),
        ('bbpVersion', (u16 * 2)),
        ('mp_parentSize', u16),
        ('mp_childSize', u16),
        ('mp_parentMaxSize', u16),
        ('mp_childMaxSize', u16),
        ('mp_sendSize', u16),
        ('mp_recvSize', u16),
        ('mp_maxSendSize', u16),
        ('mp_maxRecvSize', u16),
        ('mp_parentVCount', u16),
        ('mp_childVCount', u16),
        ('mp_parentInterval', u16),
        ('mp_childInterval', u16),
        ('mp_parentIntervalTick', OSTick),
        ('mp_childIntervalTick', OSTick),
        ('mp_minFreq', u16),
        ('mp_freq', u16),
        ('mp_maxFreq', u16),
        ('mp_vsyncOrderedFlag', u16),
        ('mp_vsyncFlag', u16),
        ('mp_count', s16),
        ('mp_limitCount', s16),
        ('mp_resumeFlag', u16),
        ('mp_prevPollBitmap', u16),
        ('mp_prevWmHeader', u16),
        ('mp_prevTxop', u16),
        ('mp_prevDataLength', u16),
        ('mp_recvBufSel', u16),
        ('mp_recvBufSize', u16),
        ('mp_recvBuf', (POINTER(WMMpRecvBuf) * 2)),
        ('mp_sendBuf', u32), #POINTER(u32)),
        ('mp_sendBufSize', u16),
        ('mp_ackTime', u16),
        ('mp_waitAckFlag', u16),
        ('mp_readyBitmap', u16),
        ('mp_newFrameFlag', u16),
        ('mp_setDataFlag', u16),
        ('mp_sentDataFlag', u16),
        ('mp_bufferEmptyFlag', u16),
        ('mp_isPolledFlag', u16),
        ('mp_minPollBmpMode', u16),
        ('mp_singlePacketMode', u16),
        ('mp_fixFreqMode', u16),
        ('mp_defaultRetryCount', u16),
        ('mp_ignoreFatalErrorMode', u16),
        ('mp_ignoreSizePrecheckMode', u16),
        ('mp_pingFlag', u16),
        ('mp_pingCounter', u16),
        ('dcf_destAdr', (u8 * 6)),
        ('dcf_sendData', u32), #POINTER(u16)),
        ('dcf_sendSize', u16),
        ('dcf_recvBufSel', u16),
        ('dcf_recvBuf', (POINTER(WMDcfRecvBuf) * 2)),
        ('dcf_recvBufSize', u16),
        ('curr_tgid', u16),
        ('linkLevel', u16),
        ('minRssi', u16),
        ('rssiCounter', u16),
        ('beaconIndicateFlag', u16),
        ('wepKeyId', u16),
        ('pwrMgtMode', u16),
        ('reserved_d', (u8 * 4)),
        ('VSyncBitmap', u16),
        ('valarm_queuedFlag', u16),
        ('v_tsf', u32),
        ('v_tsf_bak', u32),
        ('v_remain', u32),
        ('valarm_counter', u16),
        ('reserved_e', (u8 * 2)),
        ('MacAddress', (u8 * 6)),
        ('mode', u16),
        ('pparam', WMParentParam),
        ('childMacAddress', ((u8 * 6) * 15)),
        ('child_bitmap', u16),
        ('pInfoBuf', u32), #POINTER(WMBssDesc)),
        ('aid', u16),
        ('parentMacAddress', (u8 * 6)),
        ('scan_channel', u16),
        ('reserved_f', (u8 * 4)),
        ('wepMode', u16),
        ('wep_flag', BOOL),
        ('wepKey', (u16 * 40)),
        ('rate', u16),
        ('preamble', u16),
        ('tmptt', u16),
        ('retryLimit', u16),
        ('enableChannel', u16),
        ('allowedChannel', u16),
        ('portSeqNo', ((u16 * 8) * 16)),
        ('sendQueueData', (WMPortSendQueueData * 32)),
        ('sendQueueFreeList', WMPortSendQueue),
        ('sendQueue', (WMPortSendQueue * 4)),
        ('readyQueue', (WMPortSendQueue * 4)),
        ('sendQueueMutex', OSMutex),
        ('sendQueueInUse', BOOL),
        ('mp_lastRecvTick', (OSTick * 16)),
        ('mp_lifeTimeTick', OSTick),
    ]

class WMArm7Buf(Structure):
    _fields_ = [
        ('status', u32), #POINTER(WMStatus)),
        ('reserved_a', (u8 * 4)),
        ('fifo7to9', u32), #POINTER(u32)),
        ('reserved_b', (u8 * 4)),
        ('connectPInfo', WMBssDesc),
        ('requestBuf', (u32 * 128)),
    ]

class WMArm9Buf(Structure):
    _fields_ = [
        ('WM7', u32), #POINTER(WMArm7Buf)),
        ('status', u32), #POINTER(WMStatus)),
        ('indbuf', u32), #POINTER(u32)),
        ('fifo9to7', u32), #POINTER(u32)),
        ('fifo7to9', u32), #POINTER(u32)),
        ('dmaNo', u16),
        ('scanOnlyFlag', u16),
        ('CallbackTable', (WMCallbackFunc * 44)),
        ('indCallback', WMCallbackFunc),
        ('portCallbackTable', (WMCallbackFunc * 16)),
        ('portCallbackArgument', (u32 * 16)),
        ('connectedAidBitmap', u32),
        ('myAid', u16),
        ('reserved1', (u8 * 174)),
    ]

class WMStartScanReq(Structure):
    _fields_ = [
        ('apiid', u16),
        ('channel', u16),
        ('scanBuf', u32), #POINTER(WMBssDesc)),
        ('maxChannelTime', u16),
        ('bssid', (u8 * 6)),
    ]

class WMStartScanExReq(Structure):
    _fields_ = [
        ('apiid', u16),
        ('channelList', u16),
        ('scanBuf', u32), #POINTER(WMBssDesc)),
        ('scanBufSize', u16),
        ('maxChannelTime', u16),
        ('bssid', (u8 * 6)),
        ('scanType', u16),
        ('ssidLength', u16),
        ('ssid', (u8 * 32)),
        ('ssidMatchLength', u16),
        ('rsv', (u16 * 2)),
    ]

class WMStartConnectReq(Structure):
    _fields_ = [
        ('apiid', u16),
        ('reserved', u16),
        ('pInfo', u32), #POINTER(WMBssDesc)),
        ('ssid', (u8 * 24)),
        ('powerSave', BOOL),
        ('reserved2', u16),
        ('authMode', u16),
    ]

class WMMeasureChannelReq(Structure):
    _fields_ = [
        ('apiid', u16),
        ('ccaMode', u16),
        ('edThreshold', u16),
        ('channel', u16),
        ('measureTime', u16),
    ]

class WMSetMPParameterReq(Structure):
    _fields_ = [
        ('apiid', u16),
        ('reserved', u16),
        ('param', WMMPParam),
    ]

class WMStartMPReq(Structure):
    _fields_ = [
        ('apiid', u16),
        ('rsv1', u16),
        ('recvBuf', u32), #POINTER(u32)),
        ('recvBufSize', u32),
        ('sendBuf', u32), #POINTER(u32)),
        ('sendBufSize', u32),
        ('param', WMMPParam),
    ]

class WMStartTestModeReq(Structure):
    _fields_ = [
        ('apiid', u16),
        ('control', u16),
        ('signal', u16),
        ('rate', u16),
        ('channel', u16),
    ]

class WMStartTestRxModeReq(Structure):
    _fields_ = [
        ('apiid', u16),
        ('channel', u16),
    ]

class WMCallback(Structure):
    _fields_ = [
        ('apiid', u16),
        ('errcode', u16),
        ('wlCmdID', u16),
        ('wlResult', u16),
    ]

class WMStartParentCallback(Structure):
    _fields_ = [
        ('apiid', u16),
        ('errcode', u16),
        ('wlCmdID', u16),
        ('wlResult', u16),
        ('state', u16),
        ('macAddress', (u8 * 6)),
        ('aid', u16),
        ('reason', u16),
        ('ssid', (u8 * 24)),
        ('parentSize', u16),
        ('childSize', u16),
    ]

class WMStartScanCallback(Structure):
    _fields_ = [
        ('apiid', u16),
        ('errcode', u16),
        ('wlCmdID', u16),
        ('wlResult', u16),
        ('state', u16),
        ('macAddress', (u8 * 6)),
        ('channel', u16),
        ('linkLevel', u16),
        ('ssidLength', u16),
        ('ssid', (u16 * 16)),
        ('gameInfoLength', u16),
        ('gameInfo', WMGameInfo),
    ]

class WMStartScanExCallback(Structure):
    _fields_ = [
        ('apiid', u16),
        ('errcode', u16),
        ('wlCmdID', u16),
        ('wlResult', u16),
        ('state', u16),
        ('channelList', u16),
        ('reserved', (u8 * 2)),
        ('bssDescCount', u16),
        ('bssDesc', (POINTER(WMBssDesc) * 16)),
        ('linkLevel', (u16 * 16)),
    ]

class WMStartConnectCallback(Structure):
    _fields_ = [
        ('apiid', u16),
        ('errcode', u16),
        ('wlCmdID', u16),
        ('wlResult', u16),
        ('state', u16),
        ('aid', u16),
        ('reason', u16),
        ('wlStatus', u16),
        ('macAddress', (u8 * 6)),
        ('parentSize', u16),
        ('childSize', u16),
    ]

class WMDisconnectCallback(Structure):
    _fields_ = [
        ('apiid', u16),
        ('errcode', u16),
        ('wlCmdID', u16),
        ('wlResult', u16),
        ('tryBitmap', u16),
        ('disconnectedBitmap', u16),
    ]

class WMSetMPParameterCallback(Structure):
    _fields_ = [
        ('apiid', u16),
        ('errcode', u16),
        ('mask', u32),
        ('oldParam', WMMPParam),
    ]

class WMStartMPCallback(Structure):
    _fields_ = [
        ('apiid', u16),
        ('errcode', u16),
        ('state', u16),
        ('reserved', (u8 * 2)),
        ('recvBuf', u32), #POINTER(WMMpRecvBuf)),
        ('timeStamp', u16),
        ('rate_rssi', u16),
        ('destAdrs', (u8 * 6)),
        ('srcAdrs', (u8 * 6)),
        ('seqNum', u16),
        ('tmptt', u16),
        ('pollbmp', u16),
        ('reserved2', u16),
    ]

class WMStartDCFCallback(Structure):
    _fields_ = [
        ('apiid', u16),
        ('errcode', u16),
        ('state', u16),
        ('reserved', (u8 * 2)),
        ('recvBuf', u32), #POINTER(WMDcfRecvBuf)),
    ]

class WMMeasureChannelCallback(Structure):
    _fields_ = [
        ('apiid', u16),
        ('errcode', u16),
        ('wlCmdID', u16),
        ('wlResult', u16),
        ('channel', u16),
        ('ccaBusyRatio', u16),
    ]

class WMGetWirelessCounterCallback(Structure):
    _fields_ = [
        ('apiid', u16),
        ('errcode', u16),
        ('wlCmdID', u16),
        ('wlResult', u16),
        ('TX_Success', u32),
        ('TX_Failed', u32),
        ('TX_Retry', u32),
        ('TX_AckError', u32),
        ('TX_Unicast', u32),
        ('TX_Multicast', u32),
        ('TX_WEP', u32),
        ('TX_Beacon', u32),
        ('RX_RTS', u32),
        ('RX_Fragment', u32),
        ('RX_Unicast', u32),
        ('RX_Multicast', u32),
        ('RX_WEP', u32),
        ('RX_Beacon', u32),
        ('RX_FCSError', u32),
        ('RX_DuplicateError', u32),
        ('RX_MPDuplicateError', u32),
        ('RX_ICVError', u32),
        ('RX_FrameCtrlError', u32),
        ('RX_LengthError', u32),
        ('RX_PLCPError', u32),
        ('RX_BufferOverflowError', u32),
        ('RX_PathError', u32),
        ('RX_RateError', u32),
        ('RX_FCSOK', u32),
        ('TX_MP', u32),
        ('TX_KeyData', u32),
        ('TX_NullKey', u32),
        ('RX_MP', u32),
        ('RX_MPACK', u32),
        ('MPKeyResponseError', (u32 * 15)),
    ]

class WMIndCallback(Structure):
    _fields_ = [
        ('apiid', u16),
        ('errcode', u16),
        ('state', u16),
        ('reason', u16),
    ]

class WMPortSendCallback(Structure):
    _fields_ = [
        ('apiid', u16),
        ('errcode', u16),
        ('wlCmdID', u16),
        ('wlResult', u16),
        ('state', u16),
        ('port', u16),
        ('destBitmap', u16),
        ('restBitmap', u16),
        ('sentBitmap', u16),
        ('rsv', u16),
        ('data', u32), #POINTER(u16)),
        ('seqNo', u16),
        ('callback', WMCallbackFunc),
        ('arg', u32),
        ('maxSendDataSize', u16),
        ('maxRecvDataSize', u16),
    ]

class union__anonymous_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_wm_common_wm_h_2005_5_(Union):
    _fields_ = [
        ('size', u16),
        ('length', u16),
    ]

class WMPortRecvCallback(Structure):
    _fields_ = [
        ('apiid', u16),
        ('errcode', u16),
        ('state', u16),
        ('port', u16),
        ('recvBuf', u32), #POINTER(WMMpRecvBuf)),
        ('data', u32), #POINTER(u16)),
        ('length', u16),
        ('aid', u16),
        ('macAddress', (u8 * 6)),
        ('seqNo', u16),
        ('arg', u32),
        ('myAid', u16),
        ('connectedAidBitmap', u16),
        ('ssid', (u8 * 24)),
        ('reason', u16),
        ('rsv', u16),
        ('maxSendDataSize', u16),
        ('maxRecvDataSize', u16),
    ]

class WMBeaconRecvIndCallback(Structure):
    _fields_ = [
        ('apiid', u16),
        ('errcode', u16),
        ('state', u16),
        ('tgid', u16),
        ('wmstate', u16),
        ('gameInfoLength', u16),
        ('gameInfo', WMGameInfo),
    ]

class WMStartTestModeCallback(Structure):
    _fields_ = [
        ('apiid', u16),
        ('errcode', u16),
        ('RFadr5', u32),
        ('RFadr6', u32),
        ('PllLockCheck', u16),
        ('RFMDflag', u16),
    ]

class WMStopTestRxModeCallback(Structure):
    _fields_ = [
        ('apiid', u16),
        ('errcode', u16),
        ('fcsOk', u32),
        ('fcsErr', u32),
    ]
