from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.chat import *

CALLBACK_RAW = 0
CALLBACK_DISCONNECTED = 1
CALLBACK_PRIVATE_MESSAGE = 2
CALLBACK_INVITED = 3
CALLBACK_CHANNEL_MESSAGE = 4
CALLBACK_KICKED = 5
CALLBACK_USER_JOINED = 6
CALLBACK_USER_PARTED = 7
CALLBACK_USER_CHANGED_NICK = 8
CALLBACK_TOPIC_CHANGED = 9
CALLBACK_CHANNEL_MODE_CHANGED = 10
CALLBACK_USER_MODE_CHANGED = 11
CALLBACK_USER_LIST_UPDATED = 12
CALLBACK_ENUM_CHANNELS_EACH = 13
CALLBACK_ENUM_CHANNELS_ALL = 14
CALLBACK_ENTER_CHANNEL = 15
CALLBACK_GET_CHANNEL_TOPIC = 16
CALLBACK_GET_CHANNEL_MODE = 17
CALLBACK_GET_CHANNEL_PASSWORD = 18
CALLBACK_ENUM_USERS = 19
CALLBACK_GET_USER_INFO = 20
CALLBACK_GET_BASIC_USER_INFO = 21
CALLBACK_GET_CHANNEL_BASIC_USER_INFO = 22
CALLBACK_GET_USER_MODE = 23
CALLBACK_ENUM_CHANNEL_BANS = 24
CALLBACK_NICK_ERROR = 25
CALLBACK_CHANGE_NICK = 26
CALLBACK_NEW_USER_LIST = 27
CALLBACK_BROADCAST_KEY_CHANGED = 28
CALLBACK_GET_GLOBAL_KEYS = 29
CALLBACK_GET_CHANNEL_KEYS = 30
CALLBACK_AUTHENTICATE_CDKEY = 31
CALLBACK_GET_UDPRELAY = 32
CALLBACK_NUM = 33


class ciCallbackRawParams(Structure):
    _fields_ = [
        ('raw', POINTER32(c_char)),
    ]

class ciCallbackDisconnectedParams(Structure):
    _fields_ = [
        ('reason', POINTER32(c_char)),
    ]

class ciCallbackPrivateMessageParams(Structure):
    _fields_ = [
        ('user', POINTER32(c_char)),
        ('message', POINTER32(c_char)),
        ('type', c_int),
    ]

class ciCallbackInvitedParams(Structure):
    _fields_ = [
        ('channel', POINTER32(c_char)),
        ('user', POINTER32(c_char)),
    ]

class ciCallbackChannelMessageParams(Structure):
    _fields_ = [
        ('channel', POINTER32(c_char)),
        ('user', POINTER32(c_char)),
        ('message', POINTER32(c_char)),
        ('type', c_int),
    ]

class ciCallbackKickedParams(Structure):
    _fields_ = [
        ('channel', POINTER32(c_char)),
        ('user', POINTER32(c_char)),
        ('reason', POINTER32(c_char)),
    ]

class ciCallbackUserJoinedParams(Structure):
    _fields_ = [
        ('channel', POINTER32(c_char)),
        ('user', POINTER32(c_char)),
        ('mode', c_int),
    ]

class ciCallbackUserPartedParams(Structure):
    _fields_ = [
        ('channel', POINTER32(c_char)),
        ('user', POINTER32(c_char)),
        ('why', c_int),
        ('reason', POINTER32(c_char)),
        ('kicker', POINTER32(c_char)),
    ]

class ciCallbackUserChangedNickParams(Structure):
    _fields_ = [
        ('channel', POINTER32(c_char)),
        ('oldNick', POINTER32(c_char)),
        ('newNick', POINTER32(c_char)),
    ]

class ciCallbackTopicChangedParams(Structure):
    _fields_ = [
        ('channel', POINTER32(c_char)),
        ('topic', POINTER32(c_char)),
    ]

class ciCallbackChannelModeChangedParams(Structure):
    _fields_ = [
        ('channel', POINTER32(c_char)),
        ('mode', POINTER32(CHATChannelMode)),
    ]

class ciCallbackUserModeChangedParams(Structure):
    _fields_ = [
        ('channel', POINTER32(c_char)),
        ('user', POINTER32(c_char)),
        ('mode', c_int),
    ]

class ciCallbackUserListUpdatedParams(Structure):
    _fields_ = [
        ('channel', POINTER32(c_char)),
    ]

class ciCallbackConnectParams(Structure):
    _fields_ = [
        ('success', CHATBool),
    ]

class ciCallbackEnumChannelsEachParams(Structure):
    _fields_ = [
        ('success', CHATBool),
        ('index', c_int),
        ('channel', POINTER32(c_char)),
        ('topic', POINTER32(c_char)),
        ('numUsers', c_int),
        ('param', c_void_p32),
    ]

class ciCallbackEnumChannelsAllParams(Structure):
    _fields_ = [
        ('success', CHATBool),
        ('numChannels', c_int),
        ('channels', POINTER32(POINTER32(c_char))),
        ('topics', POINTER32(POINTER32(c_char))),
        ('numUsers', POINTER32(c_int)),
    ]

class ciCallbackEnterChannelParams(Structure):
    _fields_ = [
        ('success', CHATBool),
        ('result', CHATEnterResult),
        ('channel', POINTER32(c_char)),
    ]

class ciCallbackGetChannelTopicParams(Structure):
    _fields_ = [
        ('success', CHATBool),
        ('channel', POINTER32(c_char)),
        ('topic', POINTER32(c_char)),
    ]

class ciCallbackGetChannelModeParams(Structure):
    _fields_ = [
        ('success', CHATBool),
        ('channel', POINTER32(c_char)),
        ('mode', POINTER32(CHATChannelMode)),
    ]

class ciCallbackGetChannelPasswordParams(Structure):
    _fields_ = [
        ('success', CHATBool),
        ('channel', POINTER32(c_char)),
        ('enabled', CHATBool),
        ('password', POINTER32(c_char)),
    ]

class ciCallbackEnumUsersParams(Structure):
    _fields_ = [
        ('success', CHATBool),
        ('channel', POINTER32(c_char)),
        ('numUsers', c_int),
        ('users', POINTER32(POINTER32(c_char))),
        ('modes', POINTER32(c_int)),
    ]

class ciCallbackGetUserInfoParams(Structure):
    _fields_ = [
        ('success', CHATBool),
        ('nick', POINTER32(c_char)),
        ('user', POINTER32(c_char)),
        ('name', POINTER32(c_char)),
        ('address', POINTER32(c_char)),
        ('numChannels', c_int),
        ('channels', POINTER32(POINTER32(c_char))),
    ]

class ciCallbackGetBasicUserInfoParams(Structure):
    _fields_ = [
        ('success', CHATBool),
        ('nick', POINTER32(c_char)),
        ('user', POINTER32(c_char)),
        ('address', POINTER32(c_char)),
    ]

class ciCallbackGetChannelBasicUserInfoParams(Structure):
    _fields_ = [
        ('success', CHATBool),
        ('channel', POINTER32(c_char)),
        ('nick', POINTER32(c_char)),
        ('user', POINTER32(c_char)),
        ('address', POINTER32(c_char)),
    ]

class ciCallbackGetUserModeParams(Structure):
    _fields_ = [
        ('success', CHATBool),
        ('channel', POINTER32(c_char)),
        ('user', POINTER32(c_char)),
        ('mode', c_int),
    ]

class ciCallbackEnumChannelBansParams(Structure):
    _fields_ = [
        ('success', CHATBool),
        ('channel', POINTER32(c_char)),
        ('numBans', c_int),
        ('bans', POINTER32(POINTER32(c_char))),
    ]

class ciCallbackNickErrorParams(Structure):
    _fields_ = [
        ('type', c_int),
        ('nick', POINTER32(c_char)),
        ('numSuggestedNicks', c_int),
        ('suggestedNicks', POINTER32(POINTER32(c_char))),
    ]

class ciCallbackChangeNickParams(Structure):
    _fields_ = [
        ('success', CHATBool),
        ('oldNick', POINTER32(c_char)),
        ('newNick', POINTER32(c_char)),
    ]

class ciCallbackNewUserListParams(Structure):
    _fields_ = [
        ('channel', POINTER32(c_char)),
        ('numUsers', c_int),
        ('users', POINTER32(POINTER32(c_char))),
        ('modes', POINTER32(c_int)),
    ]

class ciCallbackBroadcastKeyChangedParams(Structure):
    _fields_ = [
        ('channel', POINTER32(c_char)),
        ('user', POINTER32(c_char)),
        ('key', POINTER32(c_char)),
        ('value', POINTER32(c_char)),
    ]

class ciCallbackGetGlobalKeysParams(Structure):
    _fields_ = [
        ('success', CHATBool),
        ('user', POINTER32(c_char)),
        ('num', c_int),
        ('keys', POINTER32(POINTER32(c_char))),
        ('values', POINTER32(POINTER32(c_char))),
    ]

class ciCallbackGetChannelKeysParams(Structure):
    _fields_ = [
        ('success', CHATBool),
        ('channel', POINTER32(c_char)),
        ('user', POINTER32(c_char)),
        ('num', c_int),
        ('keys', POINTER32(POINTER32(c_char))),
        ('values', POINTER32(POINTER32(c_char))),
    ]

class ciCallbackAuthenticateCDKeyParams(Structure):
    _fields_ = [
        ('result', c_int),
        ('message', POINTER32(c_char)),
    ]

class ciCallbackGetUdpRelayParams(Structure):
    _fields_ = [
        ('channel', POINTER32(c_char)),
        ('udpIp', POINTER32(c_char)),
        ('udpPort', c_short),
        ('udpKey', c_int),
    ]
