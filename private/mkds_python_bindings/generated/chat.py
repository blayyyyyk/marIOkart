from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
CHATFalse = 0
CHATTrue = 1

CHATFalse = 0
CHATTrue = 1

CHATEnterSuccess = 0
CHATBadChannelName = 1
CHATChannelIsFull = 2
CHATInviteOnlyChannel = 3
CHATBannedFromChannel = 4
CHATBadChannelPassword = 5
CHATTooManyChannels = 6
CHATEnterTimedOut = 7
CHATBadChannelMask = 8

CHATEnterSuccess = 0
CHATBadChannelName = 1
CHATChannelIsFull = 2
CHATInviteOnlyChannel = 3
CHATBannedFromChannel = 4
CHATBadChannelPassword = 5
CHATTooManyChannels = 6
CHATEnterTimedOut = 7
CHATBadChannelMask = 8

CHATFalse = 0
CHATTrue = 1

CHATFalse = 0
CHATTrue = 1

CHATEnterSuccess = 0
CHATBadChannelName = 1
CHATChannelIsFull = 2
CHATInviteOnlyChannel = 3
CHATBannedFromChannel = 4
CHATBadChannelPassword = 5
CHATTooManyChannels = 6
CHATEnterTimedOut = 7
CHATBadChannelMask = 8

CHATEnterSuccess = 0
CHATBadChannelName = 1
CHATChannelIsFull = 2
CHATInviteOnlyChannel = 3
CHATBannedFromChannel = 4
CHATBadChannelPassword = 5
CHATTooManyChannels = 6
CHATEnterTimedOut = 7
CHATBadChannelMask = 8

CHATFalse = 0
CHATTrue = 1

CHATFalse = 0
CHATTrue = 1

CHATEnterSuccess = 0
CHATBadChannelName = 1
CHATChannelIsFull = 2
CHATInviteOnlyChannel = 3
CHATBannedFromChannel = 4
CHATBadChannelPassword = 5
CHATTooManyChannels = 6
CHATEnterTimedOut = 7
CHATBadChannelMask = 8

CHATEnterSuccess = 0
CHATBadChannelName = 1
CHATChannelIsFull = 2
CHATInviteOnlyChannel = 3
CHATBannedFromChannel = 4
CHATBadChannelPassword = 5
CHATTooManyChannels = 6
CHATEnterTimedOut = 7
CHATBadChannelMask = 8

CHATFalse = 0
CHATTrue = 1

CHATFalse = 0
CHATTrue = 1

CHATEnterSuccess = 0
CHATBadChannelName = 1
CHATChannelIsFull = 2
CHATInviteOnlyChannel = 3
CHATBannedFromChannel = 4
CHATBadChannelPassword = 5
CHATTooManyChannels = 6
CHATEnterTimedOut = 7
CHATBadChannelMask = 8

CHATEnterSuccess = 0
CHATBadChannelName = 1
CHATChannelIsFull = 2
CHATInviteOnlyChannel = 3
CHATBannedFromChannel = 4
CHATBadChannelPassword = 5
CHATTooManyChannels = 6
CHATEnterTimedOut = 7
CHATBadChannelMask = 8

CHATFalse = 0
CHATTrue = 1

CHATFalse = 0
CHATTrue = 1

CHATEnterSuccess = 0
CHATBadChannelName = 1
CHATChannelIsFull = 2
CHATInviteOnlyChannel = 3
CHATBannedFromChannel = 4
CHATBadChannelPassword = 5
CHATTooManyChannels = 6
CHATEnterTimedOut = 7
CHATBadChannelMask = 8

CHATEnterSuccess = 0
CHATBadChannelName = 1
CHATChannelIsFull = 2
CHATInviteOnlyChannel = 3
CHATBannedFromChannel = 4
CHATBadChannelPassword = 5
CHATTooManyChannels = 6
CHATEnterTimedOut = 7
CHATBadChannelMask = 8

CHATFalse = 0
CHATTrue = 1

CHATFalse = 0
CHATTrue = 1

CHATEnterSuccess = 0
CHATBadChannelName = 1
CHATChannelIsFull = 2
CHATInviteOnlyChannel = 3
CHATBannedFromChannel = 4
CHATBadChannelPassword = 5
CHATTooManyChannels = 6
CHATEnterTimedOut = 7
CHATBadChannelMask = 8

CHATEnterSuccess = 0
CHATBadChannelName = 1
CHATChannelIsFull = 2
CHATInviteOnlyChannel = 3
CHATBannedFromChannel = 4
CHATBadChannelPassword = 5
CHATTooManyChannels = 6
CHATEnterTimedOut = 7
CHATBadChannelMask = 8

CHATFalse = 0
CHATTrue = 1

CHATFalse = 0
CHATTrue = 1

CHATEnterSuccess = 0
CHATBadChannelName = 1
CHATChannelIsFull = 2
CHATInviteOnlyChannel = 3
CHATBannedFromChannel = 4
CHATBadChannelPassword = 5
CHATTooManyChannels = 6
CHATEnterTimedOut = 7
CHATBadChannelMask = 8

CHATEnterSuccess = 0
CHATBadChannelName = 1
CHATChannelIsFull = 2
CHATInviteOnlyChannel = 3
CHATBannedFromChannel = 4
CHATBadChannelPassword = 5
CHATTooManyChannels = 6
CHATEnterTimedOut = 7
CHATBadChannelMask = 8


class chatChannelCallbacks(Structure):
    _fields_ = [
        ('channelMessage', chatChannelMessage),
        ('kicked', chatKicked),
        ('userJoined', chatUserJoined),
        ('userParted', chatUserParted),
        ('userChangedNick', chatUserChangedNick),
        ('topicChanged', chatTopicChanged),
        ('channelModeChanged', chatChannelModeChanged),
        ('userModeChanged', chatUserModeChanged),
        ('userListUpdated', chatUserListUpdated),
        ('newUserList', chatNewUserList),
        ('broadcastKeyChanged', chatBroadcastKeyChanged),
        ('param', c_void_p32),
    ]
chatEnumChannelBansCallback = c_void_p32
chatEnumChannelsCallbackEach = c_void_p32
chatGetChannelKeysCallback = c_void_p32
chatGetChannelModeCallback = c_void_p32
chatGetUserModeCallback = c_void_p32
chatAuthenticateCDKeyCallback = c_void_p32
chatGetChannelBasicUserInfoCallback = c_void_p32
chatGetUdpRelayCallback = c_void_p32
CHAT = c_void_p32
chatEnumChannelsCallbackAll = c_void_p32
chatEnterChannelCallback = c_void_p32
chatGetChannelPasswordCallback = c_void_p32
chatGetGlobalKeysCallback = c_void_p32
chatEnumUsersCallback = c_void_p32
chatGetChannelTopicCallback = c_void_p32
chatGetBasicUserInfoCallback = c_void_p32
chatEnumJoinedChannelsCallback = c_void_p32
chatChangeNickCallback = c_void_p32
chatGetUserInfoCallback = c_void_p32
chatChannelMessage = c_void_p32
chatTopicChanged = c_void_p32
chatUserListUpdated = c_void_p32
chatUserParted = c_void_p32
chatUserJoined = c_void_p32
chatChannelModeChanged = c_void_p32
chatUserModeChanged = c_void_p32
chatKicked = c_void_p32
chatUserChangedNick = c_void_p32
chatNewUserList = c_void_p32
chatBroadcastKeyChanged = c_void_p32
chatNickErrorCallback = c_void_p32
chatFillInUserCallback = c_void_p32

class chatGlobalCallbacks(Structure):
    _fields_ = [
        ('raw', chatRaw),
        ('disconnected', chatDisconnected),
        ('privateMessage', chatPrivateMessage),
        ('invited', chatInvited),
        ('param', c_void_p32),
    ]
chatConnectCallback = c_void_p32
CHATEnterResult = c_int

class CHATChannelMode(Structure):
    _fields_ = [
        ('InviteOnly', CHATBool),
        ('Private', CHATBool),
        ('Secret', CHATBool),
        ('Moderated', CHATBool),
        ('NoExternalMessages', CHATBool),
        ('OnlyOpsChangeTopic', CHATBool),
        ('OpsObeyChannelLimit', CHATBool),
        ('Limit', c_int),
    ]
chatDisconnected = c_void_p32
chatInvited = c_void_p32
chatPrivateMessage = c_void_p32
chatRaw = c_void_p32
CHATBool = c_int
