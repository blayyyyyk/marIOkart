from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.raceTime import *


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_struct217B488_h_4_9(Structure):
    _fields_ = [
        ('characterId', c_int),
        ('kartId', c_int),
        ('field8', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_struct217B488_h_11_9(Structure):
    _fields_ = [
        ('titleMenuSkipIntro', c_int),
        ('field4', c_int),
        ('singlePlayerMenuTarget', c_int),
        ('ghostReceive', c_int),
        ('field10', c_int),
        ('field14', c_int),
        ('unk18', c_int),
        ('field1C', c_int),
        ('driverConfigs', (struct_217B488_driver_config_t * 4)),
        ('unk50', c_int),
        ('field54', c_int),
        ('field58', c_int),
        ('playedCourses', (c_int * 5)),
        ('field70', c_int),
        ('field74', c_int),
        ('useDLResult', c_int),
        ('playerGlobalRank', c_int),
        ('gpRank', c_int),
        ('cup', c_int),
        ('ccMode', c_int),
        ('mirrorMode', c_int),
        ('playerCharacter', c_int),
        ('playerKart', c_int),
        ('courseTimes', (race_time_t * 4)),
        ('playerTotalRankPoints', c_int),
        ('driverCharacters', (c_int * 8)),
        ('driverKarts', (c_int * 8)),
        ('unkEC', (c_int * 32)),
        ('heyhoPaletteRows', (c_int * 8)),
        ('field12C', c_int),
    ]
