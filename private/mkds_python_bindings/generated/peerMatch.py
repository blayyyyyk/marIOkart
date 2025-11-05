from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.heapcommon import *
from private.mkds_python_bindings.generated.rand import *
from private.mkds_python_bindings.generated.stateMachine import *
from private.mkds_python_bindings.generated.struct217AA00 import *
from private.mkds_python_bindings.generated.types import *


class net_state_t(Structure):
    _fields_ = [
        ('heap', NNSFndHeapHandle),
        ('heapMem', c_void_p32),
        ('stateMachine', state_machine_t),
        ('raceRecvBuffers', ((u8 * 128) * 4)),
        ('profileDatagramBuffers', (net_menu_profile_dgram_t * 4)),
        ('fieldB2C', net_state_field_B2C_t),
        ('matchStatus', net_match_status_t),
        ('userData', c_int),
        ('friends', (c_int * 60)),
        ('friendStatuses', (u8 * 60)),
        ('friendListChanged', c_int),
        ('menuRecvBuffers', (net_menu_dgram_t * 4)),
        ('menuSendBuffers', (net_menu_dgram_t * 4)),
        ('field1F20', net_field_12F0_t),
        ('heapInitialized', c_int),
        ('field1F34', c_int),
        ('field1F38', c_int),
        ('frameCount', u32),
        ('field1F40', c_int),
        ('lastError', c_int),
        ('field1F48', c_int),
        ('field1F4C', u8),
        ('field1F4D', u8),
        ('field1F4E', (u8 * 4)),
        ('numConnections', u8),
        ('aidMax', u8),
        ('field1F54', u8),
        ('connectedAids', u8),
        ('newlyDisconnectedAidsBitmap', u8),
        ('lastFriendStatusFetched', u8),
    ]

class net_menu_profile_dgram_t(Structure):
    _fields_ = [
        ('header', net_menu_dgram_header_t),
        ('profile', struct_217AA00_field45C_t),
        ('field238', u32),
        ('field23C', u32),
        ('field240', u32),
    ]

class net_menu_dgram_t(Structure):
    _fields_ = [
        ('header', net_menu_dgram_header_t),
        ('data', net_menu_config_t),
    ]

class net_field_12F0_t(Structure):
    _fields_ = [
        ('gap0', (u8 * 12)),
        ('inetStatus', u32),
    ]

class net_match_status_t(Structure):
    _fields_ = [
        ('state', u32),
        ('rand', MATHRandContext32),
        ('gap1C', (u8 * 20)),
        ('field30', u32),
        ('field34', (u32 * 4)),
        ('gap44', (u8 * 5)),
        ('emblemNotSentAidMax', u8),
        ('gap4A', (u8 * 16)),
        ('receivedHellosBitmap', u8),
        ('receivedBitmap', u8),
        ('field5C', u8),
        ('field5D', u8),
        ('field5E', u8),
        ('field5F', u8),
    ]

class net_state_field_B2C_t(Structure):
    _fields_ = [
        ('control', (u8 * 3508)),
        ('fieldDB4', u8),
        ('fieldDB5', u8),
        ('gapDB6', (u8 * 2)),
        ('fieldDB8', u32),
        ('state', u32),
        ('region', net_match_property_t),
        ('matchType', net_match_property_t),
        ('fieldDF0', net_match_property_t),
        ('elo', net_match_property_t),
        ('numPlayersMatch', u8),
        ('nFriendsInMatchmaker', u8),
        ('fieldE22', u8),
        ('fieldE23', u8),
    ]

class net_menu_dgram_header_t(Structure):
    _fields_ = [
        ('opcode', u32),
        ('size', u32),
        ('aidSrc', u8),
        ('aidDest', u8),
        ('connectedAids', u8),
        ('fieldB', u8),
    ]

class net_menu_config_t(Structure):
    _fields_ = [
        ('field0', (u32 * 4)),
        ('field10', (u32 * 4)),
        ('selectedCourse', u32),
        ('field24', u32),
        ('field28', u32),
        ('vote', u32),
        ('gap30', (u8 * 4)),
        ('field34', u32),
    ]

class net_match_property_t(Structure):
    _fields_ = [
        ('value', u32),
        ('key', (c_char * 16)),
        ('field14', u8),
        ('gap15', (u8 * 2)),
        ('field16', u8),
    ]
