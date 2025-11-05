from ctypes import *
from private.mkds_python_bindings.testing.rand import *
from private.mkds_python_bindings.testing.types import *


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_struct217AA00_h_5_9_(Structure):
    _fields_ = [
        ('nickName', (u16 * 10)),
        ('emblem', (u8 * 512)),
        ('hasEmblem', u8),
        ('field216', u16),
        ('exchangeToken', c_int),
        ('unlockBits', (u8 * 4)),
        ('field228', u8),
        ('_', u8),
        ('field229', u8),
        ('field22A', u8),
        ('field22B', u8),
    ]

class struct_217AA00_field1E4C_entry_t(Structure):
    _fields_ = [
        ('unk0', (u8 * 6)),
        ('field6', u16),
    ]

class struc_252(Structure):
    _fields_ = [
        ('field0', u16),
        ('tp', u16),
        ('gap4', (u8 * 2)),
        ('flags', u8),
        ('mic', u8),
        ('field7', u8),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_struct217AA00_h_36_9_(Structure):
    _fields_ = [
        ('field0', u32),
        ('field4', u32),
        ('field8', u32),
        ('unkC', (u8 * 1062)),
        ('field432', u16),
        ('field434', u16),
        ('field436', u16),
        ('field438', u8),
        ('field439', u8),
        ('field43A', u8),
        ('field43B', u8),
        ('field43C', u8),
        ('unk43D', (u8 * 23)),
        ('field454', u16),
        ('field456', u8),
        ('field457', u8),
        ('field458', u8),
        ('unk459', (u8 * 3)),
        ('field45C', (struct_217AA00_field45C_t * 8)),
        ('unk15BC', (u8 * 12)),
        ('field15C8', u32), #POINTER(struc_252)),
        ('unk15CC', (u8 * 42)),
        ('field15F6', u8),
        ('unk15F7', (u8 * 2133)),
        ('field1E4C', (struct_217AA00_field1E4C_entry_t * 8)),
        ('field1E8C', (u16 * 8)),
        ('unk1E9C', (u8 * 20)),
        ('field1EB0', u32),
        ('unk1EB4', (u8 * 52)),
        ('field1EE8', MATHRandContext32),
    ]
