from ctypes import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.spaEmitter import *
from private.mkds_python_bindings.testing.types import *

RPTC_COLLISION_EFFECT_NONE = 0
RPTC_COLLISION_EFFECT_DIRT = 1
RPTC_COLLISION_EFFECT_SAND = 2
RPTC_COLLISION_EFFECT_GRASS = 3
RPTC_COLLISION_EFFECT_WATER = 4
RPTC_COLLISION_EFFECT_MUD = 5
RPTC_COLLISION_EFFECT_AGB_SKY_GARDEN = 6
RPTC_COLLISION_EFFECT_SNOW = 7
RPTC_COLLISION_EFFECT_FLOWER = 8

RPTC_COLLISION_EFFECT_NONE = 0
RPTC_COLLISION_EFFECT_DIRT = 1
RPTC_COLLISION_EFFECT_SAND = 2
RPTC_COLLISION_EFFECT_GRASS = 3
RPTC_COLLISION_EFFECT_WATER = 4
RPTC_COLLISION_EFFECT_MUD = 5
RPTC_COLLISION_EFFECT_AGB_SKY_GARDEN = 6
RPTC_COLLISION_EFFECT_SNOW = 7
RPTC_COLLISION_EFFECT_FLOWER = 8

RaceParticlesCollisionEffect = c_int


class rptc_rainbow_effect_t(Structure):
    _fields_ = [
        ('emitters', (POINTER(spa_emitter_t) * 3)),
    ]

class rptc_col_effect_variant_t(Structure):
    _fields_ = [
        ('emitterCount', u16),
        ('emitterIds', (u32 * 2)),
    ]

class rptc_col_effect_t(Structure):
    _fields_ = [
        ('variants', (rptc_col_effect_variant_t * 2)),
        ('func', u32),
        ('field1C', u32),
    ]

class rptc_driver_effect_controller_t(Structure):
    _fields_ = [
        ('emitterId', c_int),
        ('tireEmitterIds', (c_int * 2)),
        ('emitter', u32), #POINTER(spa_emitter_t)),
        ('tireEmitters', ((POINTER(spa_emitter_t) * 2) * 2)),
        ('field20', (POINTER(spa_emitter_t) * 2)),
        ('wallLeafEmitter', u32), #POINTER(spa_emitter_t)),
        ('bulletBillEmitter', u32), #POINTER(spa_emitter_t)),
        ('electricEmitter', u32), #POINTER(spa_emitter_t)),
        ('tireEmitterPositions', (VecFx32 * 2)),
        ('tireEmitterAxes', (VecFx16 * 2)),
    ]
