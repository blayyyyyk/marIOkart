from ctypes import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.types import *

DRAM_STATE_WAIT = 0
DRAM_STATE_SPIN_END = 1
DRAM_STATE_SPIN_START = 2
DRAM_STATE_SPINNING = 3
DRAM_STATE_SPIN_STOP = 4
DRAM_STATE_ALIGN = 5

DRAM_STATE_WAIT = 0
DRAM_STATE_SPIN_END = 1
DRAM_STATE_SPIN_START = 2
DRAM_STATE_SPINNING = 3
DRAM_STATE_SPIN_STOP = 4
DRAM_STATE_ALIGN = 5

DramState = c_int


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_race_mapobj_obstacles_dram_h_13_9_(Structure):
    _fields_ = [
        ('mobj', c_int),
        ('startStopFrameCount', u16),
        ('spinFrameCount', u16),
        ('waitFrameCount', u16),
        ('angle', u16),
        ('angularSpeed', s16),
        ('currentSpeed', fx32),
        ('startStopSpeed', fx32),
        ('speeds', (s16 * 3)),
        ('alignRemainder', s32),
    ]
