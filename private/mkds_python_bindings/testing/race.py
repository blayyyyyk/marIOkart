from ctypes import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.types import *

RACE_STATE_PRE_START = 0
RACE_STATE_STARTED = 1

RACE_STATE_PRE_START = 0
RACE_STATE_STARTED = 1

RACE_DARKENING_FOG_STATE_OFF = 0
RACE_DARKENING_FOG_STATE_ON = 1

RACE_DARKENING_FOG_STATE_OFF = 0
RACE_DARKENING_FOG_STATE_ON = 1

RaceDarkeningFogState = c_int
RaceState = c_int


class race_state_t(Structure):
    _fields_ = [
        ('state', s16),
        ('frameCounter', u32),
        ('frameCounter2', c_int),
        ('frameCounterModulo8', c_int),
        ('isOddFrame', BOOL),
        ('frameCounterModuloDriverCount', c_int),
        ('toonTableOffset', u32),
        ('toonTableUpdateCounter', u32),
        ('darkeningFogState', RaceDarkeningFogState),
        ('prevDarkeningFogState', RaceDarkeningFogState),
        ('isCamAnimMode', BOOL),
        ('isCamAnimSingleScreen', BOOL),
        ('field30', u32),
        ('field34', u32),
        ('isAwardStaffRoll', BOOL),
        ('field3C', u32),
        ('light0Dir', VecFx16),
    ]
