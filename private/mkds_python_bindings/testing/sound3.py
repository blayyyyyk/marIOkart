from ctypes import *
from private.mkds_python_bindings.testing.types import *


class seq_load_info_t(Structure):
    _fields_ = [
        ('seqId', u32),
        ('bank1', u32),
        ('bank2', u32),
    ]

class seq_heap_state_t(Structure):
    _fields_ = [
        ('seqLoadInfo', seq_load_info_t),
        ('heapLevel', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_snd_sound3_h_18_9_(Structure):
    _fields_ = [
        ('seqLoadInfo', seq_load_info_t),
        ('handle', c_int),
        ('heapState', u32), #POINTER(seq_heap_state_t)),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_snd_sound3_h_25_9_(Structure):
    _fields_ = [
        ('field0', u32),
        ('state', u32),
        ('stateBackup', u32),
        ('sfxEnabled1', BOOL),
        ('sfxEnabled2', BOOL),
        ('field14', u32),
        ('field18', u32), #POINTER(c_int)),
    ]
