from ctypes import *
from private.mkds_python_bindings.testing.thread import *
from private.mkds_python_bindings.testing.types import *

bgwkr_task_t = u32


class bgwkr_queue_t(Structure):
    _fields_ = [
        ('buffer', (bgwkr_task_t * 10)),
        ('writePtr', u8),
    ]

class bgwkr_t(Structure):
    _fields_ = [
        ('taskQueue', bgwkr_queue_t),
        ('thread', OSThread),
        ('unk', (u32 * 2)),
        ('threadStack', u32), #POINTER(u32)),
        ('requestAvailable', BOOL),
    ]
