from ctypes import *
from private.mkds_python_bindings.testing.bank import *
from private.mkds_python_bindings.testing.exchannel import *
from private.mkds_python_bindings.testing.types import *


class SNDMidiChannel(Structure):
    _fields_ = [
        ('chp', u32), #POINTER(SNDExChannel)),
        ('key', u8),
        ('pad1', u8),
        ('pad2', u16),
    ]

class SNDMidiTrack(Structure):
    _fields_ = [
        ('channels', (SNDMidiChannel * 16)),
        ('mod', SNDLfoParam),
        ('sweep_pitch', s16),
        ('prgNo', u16),
        ('pitchbend', s8),
        ('porta_time', u8),
        ('volume', u8),
        ('pan', s8),
        ('expression', u8),
        ('transpose', s8),
        ('prio', u8),
        ('bendrange', u8),
        ('porta_flag', u8),
        ('porta_key', u8),
        ('attack', u8),
        ('decay', u8),
        ('sustain', u8),
        ('release', u8),
    ]

class SNDMidiPlayer(Structure):
    _fields_ = [
        ('bank', u32), #POINTER(SNDBankData)),
        ('track', (SNDMidiTrack * 16)),
        ('main_volume', u8),
        ('prio', u8),
        ('pad', u16),
    ]
