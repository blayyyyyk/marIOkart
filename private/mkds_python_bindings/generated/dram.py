from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.mapobjInstance import *
from private.mkds_python_bindings.generated.types import *

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


class dram_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
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
DramState = c_int
