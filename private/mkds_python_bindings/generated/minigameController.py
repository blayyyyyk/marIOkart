from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.types import *

MGCNT_DRIVER_STATE_ALIVE = 0
MGCNT_DRIVER_STATE_DYING = 1
MGCNT_DRIVER_STATE_DEAD = 2

MGCNT_DRIVER_STATE_ALIVE = 0
MGCNT_DRIVER_STATE_DYING = 1
MGCNT_DRIVER_STATE_DEAD = 2


class mgcnt_t(Structure):
    _fields_ = [
        ('drivers', (mgcnt_driver_t * 8)),
        ('tryStealBalloonFunc', mgcnt_try_steal_balloon_func_t),
        ('field384', mgcnt_field384_func_t),
        ('onDamageFunc', mgcnt_on_damage_func_t),
        ('onKillFunc', mgcnt_on_kill_func_t),
        ('onEndFunc', mgcnt_on_end_func_t),
        ('gap394', (u8 * 8)),
        ('applyForceToDriverBalloonsFunc', mgcnt_apply_force_to_driver_balloons_func_t),
        ('collectableShineCount', u16),
        ('timeLimit', u16),
        ('shineRunnersRound', u16),
        ('mgEndDelayCounter', s32),
        ('maxOwnedShineCount', c_int),
        ('minOwnedShineCount', c_int),
        ('winDriverTeamId', s32),
        ('lastShineMepoIdx', u16),
        ('blncntDriverEntries', c_void_p32),
    ]

class mgcnt_driver_t(Structure):
    _fields_ = [
        ('field0', u32),
        ('field4', u32),
        ('state', MgcntDriverState),
        ('position', VecFx32),
        ('field18', s32),
        ('field1C', s32),
        ('field20', u32),
        ('field24', (u32 * 8)),
        ('field44', u16),
        ('mgDriverTeamId', u16),
        ('place', u16),
        ('balloonShineCount', c_int),
        ('balloonShineInventoryCount', u16),
        ('gap52', (u8 * 18)),
        ('micInflatingCounter', s32),
        ('keyInflating', s32),
        ('isInflating', BOOL),
    ]
mgcnt_field384_func_t = c_void_p32
mgcnt_on_end_func_t = c_void_p32
mgcnt_on_kill_func_t = c_void_p32
mgcnt_apply_force_to_driver_balloons_func_t = c_void_p32
mgcnt_on_damage_func_t = c_void_p32
mgcnt_try_steal_balloon_func_t = c_void_p32
MgcntDriverState = c_int
