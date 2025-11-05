from ctypes import *
from private.mkds_python_bindings.testing.types import *


class ranktime_gp_entry_t(Structure):
    _fields_ = [
        ('courseId', u8),
        ('rankTime50cc', u16),
        ('rankTime100cc', u16),
        ('rankTime150cc', u16),
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
