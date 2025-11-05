from ctypes import *
from private.mkds_python_bindings.testing.raceTime import *
from private.mkds_python_bindings.testing.types import *


class race_driver_result_t(Structure):
    _fields_ = [
        ('totalRankPoints', u16),
        ('globalPlace', u8),
        ('place', u8),
        ('rankPoints', u8),
        ('winCount', u8),
        ('raceTime', race_time_t),
    ]

class race_team_result_t(Structure):
    _fields_ = [
        ('totalRankPoints', u16),
        ('winCount', u8),
        ('flags', u8),
    ]

class race_results_t(Structure):
    _fields_ = [
        ('driverResults', ((race_driver_result_t * 8) * 4)),
        ('totalSkillRankPoints', u32),
        ('teamResults', ((race_team_result_t * 2) * 4)),
        ('field164', u32),
    ]
