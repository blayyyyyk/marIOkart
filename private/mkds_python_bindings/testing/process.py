from ctypes import *
from private.mkds_python_bindings.testing.nnsfnd import *

process_exit_func_t = u32
process_main_func_t = u32


class process_t(Structure):
    _fields_ = [
        ('name', u32), #POINTER(c_char)),
        ('mainFunc', process_main_func_t),
        ('exitFunc', process_exit_func_t),
        ('heapHandle', NNSFndHeapHandle),
        ('prevProcess', u32), #POINTER(process_t)),
    ]
