from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.raceTime import *
from private.mkds_python_bindings.generated.rand import *
from private.mkds_python_bindings.generated.rankpoint import *
from private.mkds_python_bindings.generated.ranktimeGp import *
from private.mkds_python_bindings.generated.types import *

RSTAT_DRIVER_STATE_0 = 0
RSTAT_DRIVER_STATE_1 = 1
RSTAT_DRIVER_STATE_2 = 2
RSTAT_DRIVER_STATE_3 = 3
RSTAT_DRIVER_STATE_4 = 4

RSTAT_DRIVER_STATE_0 = 0
RSTAT_DRIVER_STATE_1 = 1
RSTAT_DRIVER_STATE_2 = 2
RSTAT_DRIVER_STATE_3 = 3
RSTAT_DRIVER_STATE_4 = 4

RSTAT_MR_RESULT_SUCCESS = 0
RSTAT_MR_RESULT_1 = 1
RSTAT_MR_RESULT_2 = 2
RSTAT_MR_RESULT_3 = 3
RSTAT_MR_RESULT_4 = 4
RSTAT_MR_RESULT_5 = 5
RSTAT_MR_RESULT_6 = 6

RSTAT_MR_RESULT_SUCCESS = 0
RSTAT_MR_RESULT_1 = 1
RSTAT_MR_RESULT_2 = 2
RSTAT_MR_RESULT_3 = 3
RSTAT_MR_RESULT_4 = 4
RSTAT_MR_RESULT_5 = 5
RSTAT_MR_RESULT_6 = 6

RSTAT_DRIVER_STATE_0 = 0
RSTAT_DRIVER_STATE_1 = 1
RSTAT_DRIVER_STATE_2 = 2
RSTAT_DRIVER_STATE_3 = 3
RSTAT_DRIVER_STATE_4 = 4

RSTAT_DRIVER_STATE_0 = 0
RSTAT_DRIVER_STATE_1 = 1
RSTAT_DRIVER_STATE_2 = 2
RSTAT_DRIVER_STATE_3 = 3
RSTAT_DRIVER_STATE_4 = 4

RSTAT_MR_RESULT_SUCCESS = 0
RSTAT_MR_RESULT_1 = 1
RSTAT_MR_RESULT_2 = 2
RSTAT_MR_RESULT_3 = 3
RSTAT_MR_RESULT_4 = 4
RSTAT_MR_RESULT_5 = 5
RSTAT_MR_RESULT_6 = 6

RSTAT_MR_RESULT_SUCCESS = 0
RSTAT_MR_RESULT_1 = 1
RSTAT_MR_RESULT_2 = 2
RSTAT_MR_RESULT_3 = 3
RSTAT_MR_RESULT_4 = 4
RSTAT_MR_RESULT_5 = 5
RSTAT_MR_RESULT_6 = 6


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_raceStatus_h_35_9(Structure):
    _fields_ = [
        ('state', RStatDriverState),
        ('lapFrameCounter', u32),
        ('field8', u32),
        ('lapTimes', (race_time_t * 5)),
        ('totalTime', race_time_t),
        ('curLap', s32),
        ('firstPlaceTime', u32),
        ('totalMilliseconds', u32),
        ('flags', u16),
        ('curCpoi', u16),
        ('lastCorrectKeyPoint', s16),
        ('curKeyPoint', s16),
        ('curCpat', u16),
        ('highestReachedLap', u16),
        ('place', u16),
        ('driverId', u16),
        ('field3CBit89', u16),
        ('skillRankPoints', s16),
        ('cpoiProgress', fx32),
        ('raceProgress', fx32),
        ('lapProgress', fx32),
        ('cpoiMask', (u32 * 16)),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_raceStatus_h_75_9(Structure):
    _fields_ = [
        ('frameCounter', u32),
        ('timeRunning', BOOL),
        ('lapTime', race_time_t),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_raceStatus_h_82_9(Structure):
    _fields_ = [
        ('time', race_status_time_t),
        ('rankTimeGp', u16),
        ('finishedDriverCount', u16),
        ('field10', u16),
        ('driverStatus', (race_driver_status_t * 8)),
        ('placeDriverIds', (s8 * 8)),
        ('safeRng', MATHRandContext32),
        ('rngSeed', u32),
        ('randomRng', MATHRandContext32),
        ('stableRng', MATHRandContext32),
        ('rankTimeGpRtt', POINTER32(ranktime_gp_t)),
        ('resultsStored', BOOL),
        ('camAnimComplete', BOOL),
        ('cpoiKeyPointProgress', POINTER32(fx32)),
        ('gap4D8', (u8 * 4)),
        ('rankPointRpt', POINTER32(rankpoint_t)),
        ('missionResult', RStatMrResult),
        ('oneDivCpatSegmentCount', fx32),
        ('oneDivNrLaps', fx32),
        ('useTimeLimit', BOOL),
        ('uncontrollable', BOOL),
        ('timeLimit', u32),
        ('field4F8', u8),
        ('field4F9', u8),
        ('mrWinDelayCounter', u16),
        ('mrLoseDelayCounter', u16),
        ('skillRankPoints', race_skill_rankpoints_t),
    ]
RStatDriverState = c_int

class race_skill_rankpoints_t(Structure):
    _fields_ = [
        ('rankTimeDeltaPoints', c_int),
        ('firstPlacePercentagePoints', c_int),
        ('startBoostPoints', c_int),
        ('powerSlidePoints', c_int),
        ('itemHitPoints', c_int),
        ('offRoadTimePoints', c_int),
        ('wallHitPoints', c_int),
        ('damagePoints', c_int),
        ('respawnPoints', c_int),
    ]
RStatMrResult = c_int
