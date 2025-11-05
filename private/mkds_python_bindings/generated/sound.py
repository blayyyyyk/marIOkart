from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.player import *
from private.mkds_python_bindings.generated.types import *

SOUND_MODE_NORMAL = 0
SOUND_MODE_WIFI = 1
SOUND_MODE_MULTIBOOT = 2

SOUND_MODE_NORMAL = 0
SOUND_MODE_WIFI = 1
SOUND_MODE_MULTIBOOT = 2


class struc_387(Structure):
    _fields_ = [
        ('field0', u8),
        ('field1', u8),
        ('seqId', u8),
    ]

class struc_380(Structure):
    _fields_ = [
        ('field0', u32),
        ('frameCounter', u32),
        ('starMusicHandle', NNSSndHandle),
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
SoundMode = c_int
