from ctypes import *
from private.mkds_python_bindings.testing.types import *

MB_COMM_PSTATE_NONE = 0
MB_COMM_PSTATE_INIT_COMPLETE = 1
MB_COMM_PSTATE_CONNECTED = 2
MB_COMM_PSTATE_DISCONNECTED = 3
MB_COMM_PSTATE_KICKED = 4
MB_COMM_PSTATE_REQ_ACCEPTED = 5
MB_COMM_PSTATE_SEND_PROCEED = 6
MB_COMM_PSTATE_SEND_COMPLETE = 7
MB_COMM_PSTATE_BOOT_REQUEST = 8
MB_COMM_PSTATE_BOOT_STARTABLE = 9
MB_COMM_PSTATE_REQUESTED = 10
MB_COMM_PSTATE_MEMBER_FULL = 11
MB_COMM_PSTATE_END = 12
MB_COMM_PSTATE_ERROR = 13
MB_COMM_PSTATE_WAIT_TO_SEND = 14
MB_COMM_PSTATE_WM_EVENT = 2147483648

MB_COMM_PSTATE_NONE = 0
MB_COMM_PSTATE_INIT_COMPLETE = 1
MB_COMM_PSTATE_CONNECTED = 2
MB_COMM_PSTATE_DISCONNECTED = 3
MB_COMM_PSTATE_KICKED = 4
MB_COMM_PSTATE_REQ_ACCEPTED = 5
MB_COMM_PSTATE_SEND_PROCEED = 6
MB_COMM_PSTATE_SEND_COMPLETE = 7
MB_COMM_PSTATE_BOOT_REQUEST = 8
MB_COMM_PSTATE_BOOT_STARTABLE = 9
MB_COMM_PSTATE_REQUESTED = 10
MB_COMM_PSTATE_MEMBER_FULL = 11
MB_COMM_PSTATE_END = 12
MB_COMM_PSTATE_ERROR = 13
MB_COMM_PSTATE_WAIT_TO_SEND = 14
MB_COMM_PSTATE_WM_EVENT = 2147483648

MB_COMM_RESPONSE_REQUEST_KICK = 0
MB_COMM_RESPONSE_REQUEST_ACCEPT = 1
MB_COMM_RESPONSE_REQUEST_DOWNLOAD = 2
MB_COMM_RESPONSE_REQUEST_BOOT = 3

MB_COMM_RESPONSE_REQUEST_KICK = 0
MB_COMM_RESPONSE_REQUEST_ACCEPT = 1
MB_COMM_RESPONSE_REQUEST_DOWNLOAD = 2
MB_COMM_RESPONSE_REQUEST_BOOT = 3

MB_ERRCODE_SUCCESS = 0
MB_ERRCODE_INVALID_PARAM = 1
MB_ERRCODE_INVALID_STATE = 2
MB_ERRCODE_INVALID_DLFILEINFO = 3
MB_ERRCODE_INVALID_BLOCK_NO = 4
MB_ERRCODE_INVALID_BLOCK_NUM = 5
MB_ERRCODE_INVALID_FILE = 6
MB_ERRCODE_INVALID_RECV_ADDR = 7
MB_ERRCODE_WM_FAILURE = 8
MB_ERRCODE_FATAL = 9
MB_ERRCODE_MAX = 10

MB_ERRCODE_SUCCESS = 0
MB_ERRCODE_INVALID_PARAM = 1
MB_ERRCODE_INVALID_STATE = 2
MB_ERRCODE_INVALID_DLFILEINFO = 3
MB_ERRCODE_INVALID_BLOCK_NO = 4
MB_ERRCODE_INVALID_BLOCK_NUM = 5
MB_ERRCODE_INVALID_FILE = 6
MB_ERRCODE_INVALID_RECV_ADDR = 7
MB_ERRCODE_WM_FAILURE = 8
MB_ERRCODE_FATAL = 9
MB_ERRCODE_MAX = 10

MBCommPState = c_int
MBCommPStateCallback = u32
MBCommResponseRequestType = c_int
MBErrCode = c_int


class MBErrorStatus(Structure):
    _fields_ = [
        ('errcode', u16),
    ]

class MBGameRegistry(Structure):
    _fields_ = [
        ('romFilePathp', u32), #POINTER(c_char)),
        ('gameNamep', u32), #POINTER(u16)),
        ('gameIntroductionp', u32), #POINTER(u16)),
        ('iconCharPathp', u32), #POINTER(c_char)),
        ('iconPalettePathp', u32), #POINTER(c_char)),
        ('ggid', u32),
        ('maxPlayerNum', u8),
        ('pad', (u8 * 3)),
        ('userParam', (u8 * 32)),
    ]

class MBIconInfo(Structure):
    _fields_ = [
        ('palette', (u16 * 16)),
        ('data', (u16 * 256)),
    ]

class MBUserInfo(Structure):
    _fields_ = [
        ('favoriteColor', u8),
        ('playerNo', u8),
        ('nameLength', u8),
        ('name', (u16 * 10)),
    ]

class MBParentBssDesc(Structure):
    _fields_ = [
        ('length', u16),
        ('rssi', u16),
        ('bssid', (u16 * 3)),
        ('ssidLength', u16),
        ('ssid', (u8 * 32)),
        ('capaInfo', u16),
        ('rateSet', struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_mb_mb_h_453_5_),
        ('beaconPeriod', u16),
        ('dtimPeriod', u16),
        ('channel', u16),
        ('cfpPeriod', u16),
        ('cfpMaxDuration', u16),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_mb_mb_h_453_5_(Structure):
    _fields_ = [
        ('basic', u16),
        ('support', u16),
    ]

class MBParam(Structure):
    _fields_ = [
        ('boot_type', u16),
        ('parent_bss_desc', MBParentBssDesc),
    ]

class MBCommRequestData(Structure):
    _fields_ = [
        ('ggid', u32),
        ('userinfo', MBUserInfo),
        ('version', u16),
        ('fileid', u8),
        ('pad', (u8 * 3)),
    ]
