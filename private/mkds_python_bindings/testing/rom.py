from ctypes import *
from private.mkds_python_bindings.testing.types import *


class CARDRomRegion(Structure):
    _fields_ = [
        ('offset', u32),
        ('length', u32),
    ]

class CARDRomHeader(Structure):
    _fields_ = [
        ('game_name', (c_char * 12)),
        ('game_code', u32),
        ('maker_code', u16),
        ('product_id', u8),
        ('device_type', u8),
        ('device_size', u8),
        ('reserved_A', (u8 * 9)),
        ('game_version', u8),
        ('property', u8),
        ('main_rom_offset', u32),
        ('main_entry_address', u32),
        ('main_ram_address', u32),
        ('main_size', u32),
        ('sub_rom_offset', u32),
        ('sub_entry_address', u32),
        ('sub_ram_address', u32),
        ('sub_size', u32),
        ('fnt', CARDRomRegion),
        ('fat', CARDRomRegion),
        ('main_ovt', CARDRomRegion),
        ('sub_ovt', CARDRomRegion),
        ('rom_param_A', (u8 * 8)),
        ('banner_offset', u32),
        ('secure_crc', u16),
        ('rom_param_B', (u8 * 2)),
        ('main_autoload_done', u32),
        ('sub_autoload_done', u32),
        ('rom_param_C', (u8 * 8)),
        ('rom_size', u32),
        ('header_size', u32),
        ('reserved_B', (u8 * 56)),
        ('logo_data', (u8 * 156)),
        ('logo_crc', u16),
        ('header_crc', u16),
    ]
