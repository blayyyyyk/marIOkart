from ctypes import *
from private.mkds_python_bindings.testing.bank import *
from private.mkds_python_bindings.testing.exchannel import *
from private.mkds_python_bindings.testing.types import *


class SNDPlayer(Structure):
    _fields_ = [
        ('active_flag', u8),
        ('prepared_flag', u8),
        ('pause_flag', u8),
        ('pad_', u8),
        ('myNo', u8),
        ('pad2_', u8),
        ('pad3_', u8),
        ('prio', u8),
        ('volume', u8),
        ('extFader', s16),
        ('tracks', (u8 * 16)),
        ('tempo', u16),
        ('tempo_ratio', u16),
        ('tempo_counter', u16),
        ('pad__', u16),
        ('bank', u32), #POINTER(SNDBankData)),
    ]

class SNDTrack(Structure):
    _fields_ = [
        ('active_flag', u8),
        ('note_wait', u8),
        ('mute_flag', u8),
        ('tie_flag', u8),
        ('note_finish_wait', u8),
        ('porta_flag', u8),
        ('cmp_flag', u8),
        ('channel_mask_flag', u8),
        ('pan_range', u8),
        ('prgNo', u16),
        ('volume', u8),
        ('volume2', u8),
        ('pitch_bend', s8),
        ('bend_range', u8),
        ('pan', s8),
        ('ext_pan', s8),
        ('extFader', s16),
        ('ext_pitch', s16),
        ('attack', u8),
        ('decay', u8),
        ('sustain', u8),
        ('release', u8),
        ('prio', u8),
        ('transpose', s8),
        ('porta_key', u8),
        ('porta_time', u8),
        ('sweep_pitch', s16),
        ('mod', SNDLfoParam),
        ('channel_mask', u16),
        ('wait', s32),
        ('base', u32), #POINTER(u8)),
        ('cur', u32), #POINTER(u8)),
        ('call_stack', (POINTER(u8) * 3)),
        ('loop_count', (u8 * 3)),
        ('call_stack_depth', u8),
        ('channel_list', u32), #POINTER(SNDExChannel)),
    ]
