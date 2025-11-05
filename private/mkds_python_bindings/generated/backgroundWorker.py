from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_backgroundWorker_h_9_9(Structure):
    _fields_ = [
        ('buffer', (bgwkr_task_t * 10)),
        ('writePtr', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_backgroundWorker_h_15_9(Structure):
    _fields_ = [
        ('taskQueue', bgwkr_queue_t),
        ('thread', c_int),
        ('unk', (c_int * 2)),
        ('threadStack', POINTER32(c_int)),
        ('requestAvailable', c_int),
    ]
bgwkr_task_t = c_void_p32
