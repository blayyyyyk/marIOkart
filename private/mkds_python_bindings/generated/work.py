from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.alarm import *
from private.mkds_python_bindings.generated.exchannel import *
from private.mkds_python_bindings.generated.seq import *
from private.mkds_python_bindings.generated.types import *


class SNDSharedWork(Structure):
    _fields_ = [
        ('finishCommandTag', vu32),
        ('playerStatus', vu32),
        ('channelStatus', vu16),
        ('captureStatus', vu16),
        ('padding', (vu32 * 5)),
        ('player', (struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_snd_common_work_h_153_5 * 16)),
        ('globalVariable', (vs16 * 16)),
    ]

class SNDSpReg(Structure):
    _fields_ = [
        ('chCtrl', (u32 * 16)),
    ]

class SNDDriverInfo(Structure):
    _fields_ = [
        ('work', SNDWork),
        ('chCtrl', (u32 * 16)),
        ('workAddress', POINTER32(SNDWork)),
        ('lockedChannels', u32),
        ('padding', (u32 * 6)),
    ]

class SNDChannelInfo(Structure):
    _fields_ = [
        ('activeFlag', BOOL),
        ('lockFlag', BOOL),
        ('volume', u16),
        ('pan', u8),
        ('pad', u8),
        ('envStatus', SNDEnvStatus),
    ]

class SNDPlayerInfo(Structure):
    _fields_ = [
        ('activeFlag', BOOL),
        ('pauseFlag', BOOL),
        ('trackBitMask', u16),
        ('tempo', u16),
        ('volume', u8),
        ('pad', u8),
        ('pad2', u16),
    ]

class SNDTrackInfo(Structure):
    _fields_ = [
        ('prgNo', u16),
        ('volume', u8),
        ('volume2', u8),
        ('pitchBend', s8),
        ('bendRange', u8),
        ('pan', u8),
        ('transpose', s8),
        ('pad', u8),
        ('chCount', u8),
        ('channel', (u8 * 16)),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_snd_common_work_h_153_5(Structure):
    _fields_ = [
        ('variable', (vs16 * 16)),
        ('tickCounter', vu32),
    ]

class SNDWork(Structure):
    _fields_ = [
        ('channel', (SNDExChannel * 16)),
        ('player', (SNDPlayer * 16)),
        ('track', (SNDTrack * 32)),
        ('alarm', (SNDAlarm * 8)),
    ]
