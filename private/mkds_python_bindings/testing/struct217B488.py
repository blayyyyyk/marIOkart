from ctypes import *
from private.mkds_python_bindings.testing.raceTime import *
from private.mkds_python_bindings.testing.types import *


class struct_217B488_driver_config_t(Structure):
    _fields_ = [
        ('characterId', u32),
        ('kartId', u32),
        ('field8', u32),
    ]

class struct_217B488_t(Structure):
    _fields_ = [
        ('titleMenuSkipIntro', u32),
        ('field4', u32),
        ('singlePlayerMenuTarget', u32),
        ('ghostReceive', BOOL),
        ('field10', u32),
        ('field14', u32),
        ('unk18', u32),
        ('field1C', u32),
        ('driverConfigs', (struct_217B488_driver_config_t * 4)),
        ('unk50', u32),
        ('field54', u8),
        ('field58', u32),
        ('playedCourses', (u32 * 5)),
        ('field70', u32),
        ('field74', u32),
        ('useDLResult', BOOL),
        ('playerGlobalRank', s32),
        ('gpRank', u32),
        ('cup', u32),
        ('ccMode', u32),
        ('mirrorMode', u32),
        ('playerCharacter', u32),
        ('playerKart', u32),
        ('courseTimes', (race_time_t * 4)),
        ('playerTotalRankPoints', u16),
        ('driverCharacters', (u32 * 8)),
        ('driverKarts', (u32 * 8)),
        ('unkEC', (u8 * 32)),
        ('heyhoPaletteRows', (u32 * 8)),
        ('field12C', u32),
    ]
