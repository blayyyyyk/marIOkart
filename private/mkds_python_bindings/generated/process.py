from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32

class process_t(Structure):
    _fields_ = [
        ('name', POINTER32(c_char)),
        ('mainFunc', process_main_func_t),
        ('exitFunc', process_exit_func_t),
        ('heapHandle', c_int),
        ('prevProcess', POINTER32(process_t)),
    ]
process_main_func_t = c_void_p32
process_exit_func_t = c_void_p32
