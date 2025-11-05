from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.mapobjInstance import *
from private.mkds_python_bindings.generated.model import *
from private.mkds_python_bindings.generated.types import *

WATER_STATE_0 = 0
WATER_STATE_1 = 1
WATER_STATE_2 = 2
WATER_STATE_3 = 3

WATER_STATE_0 = 0
WATER_STATE_1 = 1
WATER_STATE_2 = 2
WATER_STATE_3 = 3


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
        ('waterANsbmd', c_void_p32),
        ('waterCNsbmd', c_void_p32),
        ('waterAModel', model_t),
        ('waterCModel', model_t),
        ('transformMtx', MtxFx43),
        ('isDiveable', BOOL),
        ('waterEfctNsbmd', c_void_p32),
        ('waterEfctNsbca', c_void_p32),
        ('waterEfctNsbma', c_void_p32),
    ]

class water_splash_state_t(Structure):
    _fields_ = [
        ('splashY', fx32),
        ('nsbmd', c_void_p32),
        ('nsbca', c_void_p32),
        ('nsbma', c_void_p32),
    ]
WaterState = c_int
