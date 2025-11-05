from ctypes import *
from private.mkds_python_bindings.testing.mb import *
from private.mkds_python_bindings.testing.types import *

MB_BEACON_DATA_ATTR_FIXED_NORMAL = 0
MB_BEACON_DATA_ATTR_FIXED_NO_ICON = 1
MB_BEACON_DATA_ATTR_VOLAT = 2

MB_BEACON_DATA_ATTR_FIXED_NORMAL = 0
MB_BEACON_DATA_ATTR_FIXED_NO_ICON = 1
MB_BEACON_DATA_ATTR_VOLAT = 2

MB_BEACON_DATA_ATTR_FIXED_NORMAL = 0
MB_BEACON_DATA_ATTR_FIXED_NO_ICON = 1
MB_BEACON_DATA_ATTR_VOLAT = 2

MB_SEND_VOLAT_CALLBACK_TIMMING_BEFORE = 0
MB_SEND_VOLAT_CALLBACK_TIMMING_AFTER = 1

MBBeaconDataAttr = c_int
MBSendVolatCallbackFunc = u32
MbBeaconDataAttr = c_int


class MBGameInfoFixed(Structure):
    _fields_ = [
        ('icon', MBIconInfo),
        ('parent', MBUserInfo),
        ('maxPlayerNum', u8),
        ('pad', (u8 * 1)),
        ('gameName', (u16 * 48)),
        ('gameIntroduction', (u16 * 96)),
    ]

class MBGameInfoVolatile(Structure):
    _fields_ = [
        ('nowPlayerNum', u8),
        ('pad', (u8 * 1)),
        ('nowPlayerFlag', u16),
        ('changePlayerFlag', u16),
        ('member', (MBUserInfo * 15)),
        ('userVolatData', (u8 * 8)),
    ]

class MBGameInfo(Structure):
    _fields_ = [
        ('fixed', MBGameInfoFixed),
        ('volat', MBGameInfoVolatile),
        ('broadcastedPlayerFlag', u16),
        ('dataAttr', u8),
        ('seqNoFixed', u8),
        ('seqNoVolat', u8),
        ('fileNo', u8),
        ('pad', (u8 * 2)),
        ('ggid', u32),
        ('nextp', u32), #POINTER(MBGameInfo)),
    ]
