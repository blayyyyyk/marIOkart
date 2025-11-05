from ctypes import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.mapobjInstance import *
from private.mkds_python_bindings.testing.model import *
from private.mkds_python_bindings.testing.types import *

WATER_STATE_0 = 0
WATER_STATE_1 = 1
WATER_STATE_2 = 2
WATER_STATE_3 = 3

WATER_STATE_0 = 0
WATER_STATE_1 = 1
WATER_STATE_2 = 2
WATER_STATE_3 = 3

WaterState = c_int


class water_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
    ]

class water_splash_t(Structure):
    _fields_ = [
        ('mobj', mobj_inst_t),
    ]

class water_state_t(Structure):
    _fields_ = [
        ('waterAPosition', VecFx32),
        ('waterCPosition', VecFx32),
        ('basePosition', VecFx32),
        ('state', WaterState),
        ('tideAmplitude', fx32),
        ('tidePhase', u16),
        ('tideSpeed', u16),
        ('tideProgress', fx32),
        ('field34', u16),
        ('field36', u16),
        ('field38', u16),
        ('field3A', u16),
        ('waterMovePhase', u16),
        ('field3E', u16),
        ('waterMoveSpeed', u16),
        ('waterAMoveDistance', fx32),
        ('waterCMoveDistance', fx32),
        ('field4C', fx32),
        ('waterCMovePhaseDifference', u16),
        ('field52', u16),
        ('waterAFirst', BOOL),
        ('waterAAlpha', u16),
        ('waterCAlpha', u16),
        ('field5C', u16),
        ('waterANsbmd', u32),
        ('waterCNsbmd', u32),
        ('waterAModel', model_t),
        ('waterCModel', model_t),
        ('transformMtx', MtxFx43),
        ('isDiveable', BOOL),
        ('waterEfctNsbmd', u32),
        ('waterEfctNsbca', u32),
        ('waterEfctNsbma', u32),
    ]

class water_splash_state_t(Structure):
    _fields_ = [
        ('splashY', fx32),
        ('nsbmd', u32),
        ('nsbca', u32),
        ('nsbma', u32),
    ]
