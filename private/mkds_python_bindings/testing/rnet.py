from ctypes import *
from private.mkds_python_bindings.testing.stateMachine import *
from private.mkds_python_bindings.testing.struc_351 import *
from private.mkds_python_bindings.testing.tick import *
from private.mkds_python_bindings.testing.types import *


class rnet_item_action_entry_t_(Structure):
    _fields_ = [
        ('field0', u32),
        ('buffer', u32),
        ('filled', u8),
        ('item', u8),
        ('action', u8),
        ('size', u8),
    ]

class rnet_driver_state_field20_t_(Structure):
    _fields_ = [
        ('field1', u16),
        ('field2', u16),
        ('field3', u16),
        ('field4', u16),
    ]

class rnet_driver_item_action_buffer_t_(Structure):
    _fields_ = [
        ('itemActions', (u8 * 16)),
        ('itemEventData', (u8 * 48)),
    ]

class rnet_driver_state_t_(Structure):
    _fields_ = [
        ('field0', u32),
        ('raceProgress', u32),
        ('state', struc_351),
        ('field20', (rnet_driver_state_field20_t * 4)),
        ('itemActionsBuffer', rnet_driver_item_action_buffer_t),
    ]

class rnet_dgram_t_(Structure):
    _fields_ = [
        ('src', u8),
        ('dst', u8),
        ('op', u8),
        ('field1', u8),
        ('field2', u8),
        ('field3', u8),
        ('data', (u8 * 104)),
    ]

class struc_222_field_68_t_(Structure):
    _fields_ = [
        ('field0', u32),
        ('field4', u32),
        ('frameCounter', u32),
        ('fieldC', u16),
        ('fieldE', u16),
        ('field10', u32),
        ('field14', u16),
        ('gap16', (u8 * 2)),
    ]

class struc_222(Structure):
    _fields_ = [
        ('netDriverState', rnet_driver_state_t),
        ('field68', struc_222_field_68_t),
        ('status', c_int),
        ('field84', u16),
        ('field86a', u8),
        ('place', u8),
        ('field87', u8),
    ]

class rnet_packet_sync_t_(Structure):
    _fields_ = [
        ('srcAid', u16),
        ('gap2', (u8 * 2)),
        ('field4', OSTick),
        ('fieldC', OSTick),
        ('field14', OSTick),
        ('field1C', u32),
        ('field20', u32),
        ('field24', u32),
    ]

class rnet_ping_t_(Structure):
    _fields_ = [
        ('data', rnet_packet_sync_t),
        ('field28', (u32 * 2)),
        ('field30', u32),
        ('field34', u8),
        ('field35', u8),
        ('field36', u8),
        ('field37', u8),
    ]

class net_race_state_t(Structure):
    _fields_ = [
        ('stateMachine', state_machine_t),
        ('pingStatuses', (rnet_ping_t * 4)),
        ('fieldF4', u32),
        ('lastSentAid', u16),
        ('heardFromBitmap', u8),
        ('fieldFB', u8),
        ('initialAidsBitmap', u8),
        ('fieldFD', u8),
        ('gapFE', (u8 * 6)),
        ('drivers', (struc_222 * 4)),
        ('itemActionSlots', (rnet_item_action_entry_t * 16)),
        ('incomingItemActions', ((rnet_item_action_entry_t * 4) * 16)),
        ('itemActionProcessed', ((u8 * 4) * 16)),
        ('bufferAvailable', s16),
        ('gap726', (u8 * 2)),
        ('frameCounter', c_int),
        ('field72C', c_int),
        ('gap730', (u8 * 4)),
        ('lastAidSent', u16),
        ('field_736', u16),
        ('idleTime', u16),
        ('nextAid', u8),
        ('gap73B', (u8 * 3)),
        ('field73E', u8),
        ('gap73F', u8),
        ('field740', (u16 * 4)),
        ('sendBufferHeader2', u8),
        ('connectedAidsBitmap', u8),
        ('field74A', (u8 * 4)),
        ('field74E', (u8 * 4)),
        ('gap752', (u8 * 2)),
        ('dwcSendBuffer', u32), #POINTER(rnet_dgram_t)),
        ('packetNextState', u32),
        ('flags', u16),
        ('gap75E', (u8 * 1)),
        ('field75F', u8),
    ]

class rnet_aid_map_t(Structure):
    _fields_ = [
        ('driverToAid', (s8 * 4)),
        ('initialAids', u8),
        ('initialized', u8),
        ('aidToDriver', (s8 * 4)),
    ]
