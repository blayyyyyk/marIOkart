from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.player import *
from private.mkds_python_bindings.generated.types import *


class struc_379(Structure):
    _fields_ = [
        ('field0', u32),
        ('state', u32),
        ('stateBackup', u32),
        ('sfxEnabled1', BOOL),
        ('sfxEnabled2', BOOL),
        ('field14', u32),
        ('field18', POINTER32(NNSSndHandle)),
    ]

class seq_handle_t(Structure):
    _fields_ = [
        ('seqLoadInfo', seq_load_info_t),
        ('handle', NNSSndHandle),
        ('heapState', POINTER32(seq_heap_state_t)),
    ]

class seq_heap_state_t(Structure):
    _fields_ = [
        ('seqLoadInfo', seq_load_info_t),
        ('heapLevel', c_int),
    ]

class seq_load_info_t(Structure):
    _fields_ = [
        ('seqId', u32),
        ('bank1', u32),
        ('bank2', u32),
    ]
