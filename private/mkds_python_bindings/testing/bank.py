from ctypes import *
from private.mkds_python_bindings.testing.data import *
from private.mkds_python_bindings.testing.exchannel import *
from private.mkds_python_bindings.testing.types import *

SND_INST_INVALID = 0
SND_INST_PCM = 1
SND_INST_PSG = 2
SND_INST_NOISE = 3
SND_INST_DIRECTPCM = 4
SND_INST_NULL = 5
SND_INST_DRUM_SET = 16
SND_INST_KEY_SPLIT = 17

SND_INST_INVALID = 0
SND_INST_PCM = 1
SND_INST_PSG = 2
SND_INST_NOISE = 3
SND_INST_DIRECTPCM = 4
SND_INST_NULL = 5
SND_INST_DRUM_SET = 16
SND_INST_KEY_SPLIT = 17

SNDInstType = c_int


class SNDWaveArcLink(Structure):
    _fields_ = [
        ('waveArc', u32), #POINTER(SNDWaveArc)),
        ('next', u32), #POINTER(SNDWaveArcLink)),
    ]

class SNDWaveArc(Structure):
    _fields_ = [
        ('fileHeader', SNDBinaryFileHeader),
        ('blockHeader', SNDBinaryBlockHeader),
        ('topLink', u32), #POINTER(SNDWaveArcLink)),
        ('reserved', (u32 * 7)),
        ('waveCount', u32),
        ('waveOffset', (u32 * 0)),
    ]

class SNDWaveData(Structure):
    _fields_ = [
        ('param', SNDWaveParam),
        ('samples', (u8 * 0)),
    ]

class SNDBankData(Structure):
    _fields_ = [
        ('fileHeader', SNDBinaryFileHeader),
        ('blockHeader', SNDBinaryBlockHeader),
        ('waveArcLink', (SNDWaveArcLink * 4)),
        ('instCount', u32),
        ('instOffset', (u32 * 0)),
    ]

class SNDInstParam(Structure):
    _fields_ = [
        ('wave', (u16 * 2)),
        ('original_key', u8),
        ('attack', u8),
        ('decay', u8),
        ('sustain', u8),
        ('release', u8),
        ('pan', u8),
    ]

class SNDInstData(Structure):
    _fields_ = [
        ('type', u8),
        ('padding_', u8),
        ('param', SNDInstParam),
    ]

class SNDKeySplit(Structure):
    _fields_ = [
        ('key', (u8 * 8)),
        ('instOffset', (SNDInstData * 0)),
    ]

class SNDDrumSet(Structure):
    _fields_ = [
        ('min', u8),
        ('max', u8),
        ('instOffset', (SNDInstData * 0)),
    ]

class SNDInstPos(Structure):
    _fields_ = [
        ('prgNo', u32),
        ('index', u32),
    ]
