from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.raceTime import *
from private.mkds_python_bindings.generated.types import *


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_raceResults_h_5_9(Structure):
    _fields_ = [
        ('totalRankPoints', u16),
        ('globalPlace', u8),
        ('place', u8),
        ('rankPoints', u8),
        ('winCount', u8),
        ('raceTime', race_time_t),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_raceResults_h_22_9(Structure):
    _fields_ = [
        ('driverResults', ((race_driver_result_t * 8) * 4)),
        ('totalSkillRankPoints', u32),
        ('teamResults', ((race_team_result_t * 2) * 4)),
        ('field164', u32),
    ]

class race_team_result_t(Structure):
    _fields_ = [
        ('totalRankPoints', u16),
        ('winCount', u8),
        ('flags', u8),
    ]
