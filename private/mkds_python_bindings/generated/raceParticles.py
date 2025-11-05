from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.spaEmitter import *

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


class rptc_rainbow_effect_t(Structure):
    _fields_ = [
        ('emitters', (POINTER32(spa_emitter_t) * 3)),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_effects_raceParticles_h_26_9(Structure):
    _fields_ = [
        ('emitterCount', c_int),
        ('emitterIds', (c_int * 2)),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_effects_raceParticles_h_32_9(Structure):
    _fields_ = [
        ('variants', (rptc_col_effect_variant_t * 2)),
        ('BOOL', c_void_p32),
        ('field1C', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_effects_raceParticles_h_39_9(Structure):
    _fields_ = [
        ('emitterId', c_int),
        ('tireEmitterIds', (c_int * 2)),
        ('emitter', POINTER32(spa_emitter_t)),
        ('tireEmitters', ((POINTER32(spa_emitter_t) * 2) * 2)),
        ('field20', (POINTER32(spa_emitter_t) * 2)),
        ('wallLeafEmitter', POINTER32(spa_emitter_t)),
        ('bulletBillEmitter', POINTER32(spa_emitter_t)),
        ('electricEmitter', POINTER32(spa_emitter_t)),
        ('tireEmitterPositions', (c_int * 2)),
        ('tireEmitterAxes', (c_int * 2)),
    ]
RaceParticlesCollisionEffect = c_int
