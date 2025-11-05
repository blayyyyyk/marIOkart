from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.types import *


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_ranktimeGp_h_3_9(Structure):
    _fields_ = [
        ('courseId', c_int),
        ('rankTime50cc', c_int),
        ('rankTime100cc', c_int),
        ('rankTime150cc', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_ranktimeGp_h_11_9(Structure):
    _fields_ = [
        ('signature', c_int),
        ('fileSize', c_int),
        ('courseCount', c_int),
        ('rankTimeDeltaFactor', c_int),
        ('firstPlacePercentageFactor', c_int),
        ('startBoostFactor', c_int),
        ('powerSlideFactor', c_int),
        ('itemHitFactor', c_int),
        ('offRoadTimeFactor', c_int),
        ('wallHitFactor', c_int),
        ('damageFactor', c_int),
        ('respawnFactor', c_int),
        ('courseRankTimes', (ranktime_gp_entry_t * 1)),
    ]

class ranktime_gp_t(Structure):
    _fields_ = [
        ('signature', u32),
        ('fileSize', u32),
        ('courseCount', u16),
        ('rankTimeDeltaFactor', u16),
        ('firstPlacePercentageFactor', u16),
        ('startBoostFactor', u16),
        ('powerSlideFactor', u16),
        ('itemHitFactor', u16),
        ('offRoadTimeFactor', u16),
        ('wallHitFactor', u16),
        ('damageFactor', u16),
        ('respawnFactor', u16),
        ('courseRankTimes', (ranktime_gp_entry_t * 1)),
    ]

class ranktime_gp_entry_t(Structure):
    _fields_ = [
        ('courseId', u8),
        ('rankTime50cc', u16),
        ('rankTime100cc', u16),
        ('rankTime150cc', u16),
    ]
