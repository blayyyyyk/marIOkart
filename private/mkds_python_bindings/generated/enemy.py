from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.driver import *
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.mapData import *
from private.mkds_python_bindings.generated.types import *


class enemy_rubberbanding_t(Structure):
    _fields_ = [
        ('field0', c_int),
        ('driverFieldCC', c_int),
        ('rivalAggressiveness', c_int),
        ('maxDriverFieldCC', c_int),
        ('field10', c_int),
        ('place', c_int),
        ('field18', c_int),
        ('field1C', c_int),
        ('field20', c_int),
        ('field24', c_int),
        ('hasFailedStart', BOOL),
        ('hasStartBoost', BOOL),
        ('startBoostAmount', u16),
        ('field32', u16),
    ]

class enemy_item_params_t(Structure):
    _fields_ = [
        ('updateFunc', c_void_p32),
        ('field4', c_int),
        ('minTimeout', u16),
        ('timeoutRandomMax', u16),
    ]

class enemy_item_state_t(Structure):
    _fields_ = [
        ('slotItemTimeout', u16),
        ('dragItemTimeout', u16),
        ('slotItemParams', POINTER32(enemy_item_params_t)),
        ('dragItemParams', POINTER32(enemy_item_params_t)),
        ('fieldC', c_int),
        ('lxPressed', c_int),
        ('dpadUpPressed', c_int),
        ('dpadDownPressed', c_int),
        ('dpadUpCounter', u16),
        ('dpadDownCounter', u16),
        ('field20', c_int),
    ]

class enemy_field140_t(Structure):
    _fields_ = [
        ('field0', c_int),
        ('driverFieldCC', c_int),
        ('field8', c_int),
        ('fieldC', c_int),
        ('field10', c_int),
    ]

class struc_316_epoi(Structure):
    _fields_ = [
        ('nextEpoi', POINTER32(mdat_enemypoint_t)),
        ('curEpoi', POINTER32(mdat_enemypoint_t)),
        ('direction', VecFx32),
        ('areaEpoi', POINTER32(mdat_enemypoint_t)),
        ('areaEpoiValid', BOOL),
        ('driverId', u16),
        ('field1E', u8),
        ('field1F', u8),
    ]

class struc_313_mepo(Structure):
    _fields_ = [
        ('nextMepo', POINTER32(mdat_mgenemypoint_t)),
        ('curMepo', POINTER32(mdat_mgenemypoint_t)),
        ('direction', VecFx32),
        ('areaMepo', POINTER32(mdat_mgenemypoint_t)),
        ('areaMepoValid', BOOL),
    ]

class enemy_t(Structure):
    _fields_ = [
        ('driver', POINTER32(driver_t)),
        ('driverId', u16),
        ('field6', u16),
        ('epoi', struc_316_epoi),
        ('mepo', struc_313_mepo),
        ('targetPos', VecFx32),
        ('field50', VecFx32),
        ('field5C', POINTER32(VecFx32)),
        ('driftOffset', VecFx32),
        ('driftEpoiRadiusScaleUpdateCounter', u16),
        ('driftEpoiRadiusScaleUpdateFrames', u16),
        ('field70', c_int),
        ('driftEpoiRadiusScale', fx32),
        ('field78', u16),
        ('field7A', u16),
        ('field7C', c_int),
        ('field80', u16),
        ('field84', c_int),
        ('field88', c_int),
        ('field8C', c_int),
        ('driftState', c_int),
        ('driftDirection', c_int),
        ('field98', c_int),
        ('field9C', c_int),
        ('fieldA0', c_int),
        ('fieldA4', c_int),
        ('fieldA8', c_int),
        ('fieldAC', u16),
        ('targetDriver', POINTER32(driver_t)),
        ('fieldB4', c_int),
        ('fieldB8', c_int),
        ('fieldBC', c_int),
        ('targetBalloonCount', u16),
        ('balloonInflateMicTimeout', u16),
        ('isInflatingBalloon', c_int),
        ('fieldC8', u16),
        ('fieldCA', u16),
        ('fieldCC', c_int),
        ('targetShine', c_void_p32),
        ('shineIdx', u16),
        ('fieldD8', c_int),
        ('fieldDC', VecFx32),
        ('rubberbanding', enemy_rubberbanding_t),
        ('itemState', enemy_item_state_t),
        ('field140', enemy_field140_t),
        ('field154', c_int),
    ]
