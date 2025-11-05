from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.list import *
from private.mkds_python_bindings.generated.player import *
from private.mkds_python_bindings.generated.soundPool import *
from private.mkds_python_bindings.generated.types import *


class sfx_base_params_t(Structure):
    _fields_ = [
        ('maxDistance', u32),
        ('fadePart1EndDistance', u32),
        ('fadePart1EndVolume', u32),
        ('fadeStartDistance', u32),
        ('maxVolume', u32),
    ]

class sfx_params_t(Structure):
    _fields_ = [
        ('maxDistance', c_int),
        ('fadePart1EndDistance', c_int),
        ('fadePart1EndVolume', u32),
        ('fadeStartDistance', c_int),
        ('maxVolume', c_int),
        ('fadePart1Factor', fx32),
        ('fadePart2Factor', fx32),
        ('field1C', u32),
        ('maxDistanceSquare', c_int),
    ]

class snd_unkstruct_t(Structure):
    _fields_ = [
        ('field0', snd_unkstruct_field0_t),
        ('position', VecFx32),
        ('soundHandle', NNSSndHandle),
        ('field20', u32),
        ('field24', u32),
    ]

class sfx_emitter_ex_t(Structure):
    _fields_ = [
        ('emitter', sfx_emitter_t),
        ('exParams', sfx_emitter_ex_params_t),
    ]

class sfx_sound_t(Structure):
    _fields_ = [
        ('poolHandle', sp_handle_t),
        ('listLink', list_link_t),
        ('pitch', s16),
        ('sfxId', u16),
        ('seqArcId', u8),
        ('field19', u8),
        ('volume', u8),
        ('field1B', u8),
    ]

class struc_393(Structure):
    _fields_ = [
        ('elementCount', u32),
        ('elements', POINTER32(struc_394)),
    ]

class snd_unkstruct_field0_t(Structure):
    _fields_ = [
        ('sfxId', u32),
        ('position', POINTER32(VecFx32)),
        ('sfxParamsId', u32),
        ('squareDistance', c_int),
    ]

class struc_394(Structure):
    _fields_ = [
        ('field0', c_int),
        ('func', u32),
        ('arg', c_void_p32),
        ('fieldC', u32),
    ]

class sfx_emitter_t(Structure):
    _fields_ = [
        ('listLink', list_link_t),
        ('soundList', NNSFndList),
        ('field18', u32),
        ('field1C', list_link_t),
        ('position', POINTER32(VecFx32)),
        ('startFunc', c_void_p32),
        ('updateFunc', c_void_p32),
        ('field34', u16),
        ('field36', u16),
        ('field38', u32),
        ('sfxParamIdx', u8),
        ('field3D', (u8 * 3)),
        ('squareDistance', u32),
    ]

class sfx_emitter_ex_params_t(Structure):
    _fields_ = [
        ('field0', c_int),
        ('field4', c_int),
        ('pitchOffset', u32),
        ('fieldC', u32),
        ('unk10', (u8 * 12)),
    ]
