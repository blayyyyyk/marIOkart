from ctypes import *
from private.mkds_python_bindings.testing.types import *

state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_init_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32
state_machine_state_func_t = u32


class state_machine_state_t(Structure):
    _fields_ = [
        ('initFunc', state_machine_init_func_t),
        ('stateFunc', state_machine_state_func_t),
    ]

class state_machine_t(Structure):
    _fields_ = [
        ('states', u32), #POINTER(state_machine_state_t)),
        ('counter', u32),
        ('userData', u32),
        ('nrStates', u16),
        ('curState', u16),
        ('nextState', u16),
        ('gotoNextState', u16),
    ]
