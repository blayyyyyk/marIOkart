from ctypes import *
from private.mkds_python_bindings.testing.channel import *
from private.mkds_python_bindings.testing.types import *

SND_EX_CHANNEL_PCM = 0
SND_EX_CHANNEL_PSG = 1
SND_EX_CHANNEL_NOISE = 2

SND_EX_CHANNEL_PCM = 0
SND_EX_CHANNEL_PSG = 1
SND_EX_CHANNEL_NOISE = 2

SND_EX_CHANNEL_CALLBACK_DROP = 0
SND_EX_CHANNEL_CALLBACK_FINISH = 1

SND_EX_CHANNEL_CALLBACK_DROP = 0
SND_EX_CHANNEL_CALLBACK_FINISH = 1

SND_ENV_ATTACK = 0
SND_ENV_DECAY = 1
SND_ENV_SUSTAIN = 2
SND_ENV_RELEASE = 3

SND_ENV_ATTACK = 0
SND_ENV_DECAY = 1
SND_ENV_SUSTAIN = 2
SND_ENV_RELEASE = 3

SND_LFO_PITCH = 0
SND_LFO_VOLUME = 1
SND_LFO_PAN = 2

SND_LFO_PITCH = 0
SND_LFO_VOLUME = 1
SND_LFO_PAN = 2

SNDEnvStatus = c_int
SNDExChannelCallback = u32
SNDExChannelCallbackStatus = c_int
SNDExChannelType = c_int
SNDLfoTarget = c_int


class SNDWaveParam(Structure):
    _fields_ = [
        ('format', u8),
        ('loopflag', u8),
        ('rate', u16),
        ('timer', u16),
        ('loopstart', u16),
        ('looplen', u32),
    ]

class SNDLfoParam(Structure):
    _fields_ = [
        ('target', u8),
        ('speed', u8),
        ('depth', u8),
        ('range', u8),
        ('delay', u16),
    ]

class SNDLfo(Structure):
    _fields_ = [
        ('param', SNDLfoParam),
        ('delay_counter', u16),
        ('counter', u16),
    ]

class SNDExChannel(Structure):
    _fields_ = [
        ('myNo', u8),
        ('type', u8),
        ('env_status', u8),
        ('active_flag', u8),
        ('start_flag', u8),
        ('auto_sweep', u8),
        ('sync_flag', u8),
        ('pan_range', u8),
        ('original_key', u8),
        ('user_decay2', s16),
        ('key', u8),
        ('velocity', u8),
        ('init_pan', s8),
        ('user_pan', s8),
        ('user_decay', s16),
        ('user_pitch', s16),
        ('env_decay', s32),
        ('sweep_counter', s32),
        ('sweep_length', s32),
        ('attack', u8),
        ('sustain', u8),
        ('decay', u16),
        ('release', u16),
        ('prio', u8),
        ('pan', u8),
        ('volume', u16),
        ('timer', u16),
        ('lfo', SNDLfo),
        ('sweep_pitch', s16),
        ('length', s32),
        ('wave', SNDWaveParam),
        ('callback', SNDExChannelCallback),
        ('callback_data', u32),
        ('nextLink', u32), #POINTER(SNDExChannel)),
    ]

class union__anonymous_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_snd_common_exchannel_h_238_5_(Union):
    _fields_ = [
        ('data', u32),
        ('duty', SNDDuty),
    ]
