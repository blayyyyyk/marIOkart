from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
DWCi_FRIENDKEY_GENERATOR_CRC = 0
DWCi_FRIENDKEY_GENERATOR_MD5 = 1
DWCi_FRIENDKEY_GENERATOR_NUM = 2

DWCi_FRIENDKEY_GENERATOR_CRC = 0
DWCi_FRIENDKEY_GENERATOR_MD5 = 1
DWCi_FRIENDKEY_GENERATOR_NUM = 2

DWCUserData = DWCAccUserData
DWCFriendData = DWCAccFriendData

class DWCstAccFlag(Structure):
    _fields_ = [
        ('flags', c_int),
        ('reserved', c_int),
        ('reserved1', c_int),
    ]

class DWCstAccLoginId(Structure):
    _fields_ = [
        ('id_data', c_int),
        ('userid_lo32', c_int),
        ('playerid', c_int),
    ]

class DWCstAccFriendKey(Structure):
    _fields_ = [
        ('id_data', c_int),
        ('friendkey_lo32', c_int),
        ('friendkey_hi32', c_int),
    ]

class DWCstAccGsProfileId(Structure):
    _fields_ = [
        ('id_data', c_int),
        ('id', c_int),
        ('reserved', c_int),
    ]

class DWCstAccFriendData(Union):
    _fields_ = [
        ('flags', DWCAccFlag),
        ('login_id', DWCAccLoginId),
        ('friend_key', DWCAccFriendKey),
        ('gs_profile_id', DWCAccGsProfileId),
    ]

class DWCstAccUserData(Structure):
    _fields_ = [
        ('size', c_int),
        ('pseudo', DWCAccLoginId),
        ('authentic', DWCAccLoginId),
        ('gs_profile_id', c_int),
        ('flag', c_int),
        ('gamecode', c_int),
        ('reserved', (c_int * 5)),
        ('crc32', c_int),
    ]
DWCiFriendKeyGenMethod = c_int
