from ctypes import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.list import *
from private.mkds_python_bindings.testing.nnsfnd import *
from private.mkds_python_bindings.testing.soundPool import *
from private.mkds_python_bindings.testing.types import *


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

class snd_unkstruct_field0_t(Structure):
    _fields_ = [
        ('sfxId', u32),
        ('position', u32), #POINTER(VecFx32)),
        ('sfxParamsId', u32),
        ('squareDistance', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_sfx_h_36_9_(Structure):
    _fields_ = [
        ('field0', snd_unkstruct_field0_t),
        ('position', VecFx32),
        ('soundHandle', c_int),
        ('field20', u32),
        ('field24', u32),
    ]
# 2x + y = 43
# 2(8) + 12 = 44
class sfx_emitter_t(Structure):
    _fields_ = [
        ('listLink', list_link_t), # 12 0x00
        ('soundList', NNSFndList), # 12 0x0C
        ('field18', u32), # 4 0x18
        ('field1C', list_link_t), # 12 0x1C
        ('position', u32), # 4 0x28
        ('startFunc', u32), # 4 0x2C
        ('updateFunc', u32), # 4 0x30
        ('field34', c_short), # 2 0x34
        ('field38', u32), # 4 0x38
        ('sfxParamIdx', c_char), # 1
        ('squareDistance', u32), # 4 0x40
    ]

class sfx_emitter_ex_params_t(Structure):
    
    _fields_ = [
        ('field0', c_int),
        ('field4', c_int),
        ('pitchOffset', u32),
        ('fieldC', u32),
        ('unk10', (u8 * 12)),
    ]

class sfx_emitter_ex_t(Structure):
    _fields_ = [
        ('emitter', sfx_emitter_t),
        ('exParams', sfx_emitter_ex_params_t),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_sfx_h_80_9_(Structure):
    _fields_ = [
        ('poolHandle', u32), # TODO: locate "sp_handle_t"
        ('listLink', list_link_t),
        ('pitch', s16),
        ('sfxId', u16),
        ('seqArcId', u8),
        ('field19', u8),
        ('volume', u8),
        ('field1B', u8),
    ]

class struc_394(Structure):
    _fields_ = [
        ('field0', c_int),
        ('func', u32),
        ('arg', u32),
        ('fieldC', u32),
    ]

class struc_393(Structure):
    _fields_ = [
        ('elementCount', u32),
        ('elements', u32), #POINTER(struc_394)),
    ]
