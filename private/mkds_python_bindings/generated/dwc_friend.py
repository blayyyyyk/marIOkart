from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
DWC_STATUS_OFFLINE = 0
DWC_STATUS_ONLINE = 1
DWC_STATUS_PLAYING = 2
DWC_STATUS_MATCH_ANYBODY = 3
DWC_STATUS_MATCH_FRIEND = 4
DWC_STATUS_MATCH_SC_CL = 5
DWC_STATUS_MATCH_SC_SV = 6
DWC_STATUS_NUM = 7

DWC_FRIEND_STATE_INIT = 0
DWC_FRIEND_STATE_PERS_LOGIN = 1
DWC_FRIEND_STATE_LOGON = 2
DWC_FRIEND_STATE_NUM = 3

DWC_FRIEND_STATE_INIT = 0
DWC_FRIEND_STATE_PERS_LOGIN = 1
DWC_FRIEND_STATE_LOGON = 2
DWC_FRIEND_STATE_NUM = 3

DWC_BUDDY_UPDATE_STATE_WAIT = 0
DWC_BUDDY_UPDATE_STATE_CHECK = 1
DWC_BUDDY_UPDATE_STATE_PSEARCH = 2
DWC_BUDDY_UPDATE_STATE_COMPLETE = 3
DWC_BUDDY_UPDATE_STATE_NUM = 4

DWC_PERS_STATE_INIT = 0
DWC_PERS_STATE_LOGIN = 1
DWC_PERS_STATE_CONNECTED = 2
DWC_PERS_STATE_NUM = 3


class DWCunFriendDataOld(Union):
    _fields_ = [
        ('mpFriendKey', (c_int * 16)),
        ('profile', struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_dwc_base_dwc_friend_h_176_5),
        ('flags', (c_int * 16)),
    ]

class DWCstFriendControl(Structure):
    _fields_ = [
        ('state', DWCFriendState),
        ('pGpObj', POINTER32(c_int)),
        ('gpProcessCount', c_int),
        ('lastGpProcess', c_int),
        ('friendListLen', c_int),
        ('friendList', POINTER32(c_int)),
        ('buddyUpdateIdx', c_int),
        ('friendListChanged', c_int),
        ('buddyUpdateState', c_int),
        ('svUpdateComplete', c_int),
        ('persCallbackLevel', c_int),
        ('profileID', c_int),
        ('playerName', POINTER32(c_int)),
        ('updateCallback', DWCUpdateServersCallback),
        ('updateParam', c_void_p32),
        ('statusCallback', DWCFriendStatusCallback),
        ('statusParam', c_void_p32),
        ('deleteCallback', DWCDeleteFriendListCallback),
        ('deleteParam', c_void_p32),
        ('buddyCallback', DWCBuddyFriendCallback),
        ('buddyParam', c_void_p32),
        ('persLoginCallback', DWCStorageLoginCallback),
        ('persLoginParam', c_void_p32),
        ('saveCallback', DWCSaveToServerCallback),
        ('loadCallback', DWCLoadFromServerCallback),
        ('reverseBuddies', POINTER32(c_int)),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_dwc_base_dwc_friend_h_176_5(Structure):
    _fields_ = [
        ('id', c_int),
        ('pad', (c_int * 12)),
    ]
DWCFriendState = c_int
DWCFriendStatusCallback = c_void_p32
DWCLoadFromServerCallback = c_void_p32
DWCStorageLoginCallback = c_void_p32
DWCSaveToServerCallback = c_void_p32
DWCBuddyFriendCallback = c_void_p32
DWCUpdateServersCallback = c_void_p32
DWCDeleteFriendListCallback = c_void_p32
