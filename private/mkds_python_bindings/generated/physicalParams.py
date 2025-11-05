from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.types import *


class physp_t(Structure):
    _fields_ = [
        ('driverKartPhysicalParams', POINTER32(physp_kart_params_t)),
        ('driverCharPhysicalParams', POINTER32(physp_char_params_t)),
        ('field8', u32),
        ('kartPhysicalParams', POINTER32(physp_kart_params_t)),
        ('charPhysicalParams', POINTER32(physp_char_params_t)),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_drivers_physicalParams_h_3_9(Structure):
    _fields_ = [
        ('colSphereSize', c_int),
        ('colSphereZOffset', c_int),
        ('gap8', (c_int * 4)),
        ('weight', c_int),
        ('driftBoostTime', c_int),
        ('maxSpeed', c_int),
        ('baseAcceleration', c_int),
        ('field18', c_int),
        ('field1C', c_int),
        ('driftBaseAcceleration', c_int),
        ('field24', c_int),
        ('field28', c_int),
        ('deceleration', c_int),
        ('handling', c_int),
        ('drift', c_int),
        ('driftTurningCompensation', c_int),
        ('collisionVelocityMinusDirMultipliers', (c_int * 12)),
        ('collisionSpeedMultipliers', (c_int * 12)),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_drivers_physicalParams_h_29_9(Structure):
    _fields_ = [
        ('field0', c_int),
        ('weight', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_drivers_physicalParams_h_37_9(Structure):
    _fields_ = [
        ('driverKartPhysicalParams', POINTER32(physp_kart_params_t)),
        ('driverCharPhysicalParams', POINTER32(physp_char_params_t)),
        ('field8', c_int),
        ('kartPhysicalParams', POINTER32(physp_kart_params_t)),
        ('charPhysicalParams', POINTER32(physp_char_params_t)),
    ]

class physp_kart_params_t(Structure):
    _fields_ = [
        ('colSphereSize', fx32),
        ('colSphereZOffset', fx32),
        ('gap8', (u8 * 4)),
        ('weight', fx16),
        ('driftBoostTime', s16),
        ('maxSpeed', fx32),
        ('baseAcceleration', fx32),
        ('field18', fx32),
        ('field1C', fx32),
        ('driftBaseAcceleration', fx32),
        ('field24', fx32),
        ('field28', fx32),
        ('deceleration', fx32),
        ('handling', fx16),
        ('drift', fx16),
        ('driftTurningCompensation', fx32),
        ('collisionVelocityMinusDirMultipliers', (fx32 * 12)),
        ('collisionSpeedMultipliers', (fx32 * 12)),
    ]

class physp_char_params_t(Structure):
    _fields_ = [
        ('field0', fx32),
        ('weight', fx32),
    ]
