from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.darray import *
from private.mkds_python_bindings.generated.gp import *
from private.mkds_python_bindings.generated.gpiInfo import *
from private.mkds_python_bindings.generated.hashtable import *


class _GPIBuddyStatusInfo(Structure):
    _fields_ = [
        ('buddyIndex', c_int),
        ('statusState', GPEnum),
        ('richStatus', POINTER32(c_char)),
        ('gameType', POINTER32(c_char)),
        ('gameVariant', POINTER32(c_char)),
        ('gameMapName', POINTER32(c_char)),
        ('sessionFlags', c_int),
        ('buddyIp', c_int),
        ('buddyPort', c_short),
        ('hostIp', c_int),
        ('hostPrivateIp', c_int),
        ('queryPort', c_short),
        ('hostPort', c_short),
        ('quietModeFlags', GPEnum),
        ('productId', c_int),
        ('extendedInfoKeys', DArray),
    ]

class GPIProfile(Structure):
    _fields_ = [
        ('profileId', c_int),
        ('userId', c_int),
        ('buddyStatus', POINTER32(GPIBuddyStatus)),
        ('buddyStatusInfo', POINTER32(GPIBuddyStatusInfo)),
        ('cache', POINTER32(GPIInfoCache)),
        ('authSig', POINTER32(c_char)),
        ('requestCount', c_int),
        ('peerSig', POINTER32(c_char)),
    ]
gpiProfileMapFunc = c_void_p32

class GPIBuddyStatus(Structure):
    _fields_ = [
        ('buddyIndex', c_int),
        ('status', GPEnum),
        ('statusString', POINTER32(c_char)),
        ('locationString', POINTER32(c_char)),
        ('ip', c_int),
        ('port', c_short),
        ('quietModeFlags', GPEnum),
    ]

class GPIProfileList(Structure):
    _fields_ = [
        ('profileTable', HashTable),
        ('num', c_int),
        ('numBuddies', c_int),
    ]
