from ctypes import *
from private.mkds_python_bindings.testing.types import *

SOUND_MODE_NORMAL = 0
SOUND_MODE_WIFI = 1
SOUND_MODE_MULTIBOOT = 2

SOUND_MODE_NORMAL = 0
SOUND_MODE_WIFI = 1
SOUND_MODE_MULTIBOOT = 2

SoundMode = c_int


class struc_387(Structure):
    _fields_ = [
        ('field0', u8),
        ('field1', u8),
        ('seqId', u8),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_snd_sound_h_31_9_(Structure):
    _fields_ = [
        ('field0', u32),
        ('frameCounter', u32),
        ('starMusicHandle', c_int),
        ('fieldC', u32),
        ('field10', u32),
        ('lapCount', u8),
        ('curLap', u8),
        ('field16', u8),
        ('starMusicState', u8),
        ('curPlace', u8),
        ('lastSfxPlace', u8),
        ('placeChangeSfxTimeout', c_int),
        ('field20', u32),
    ]
