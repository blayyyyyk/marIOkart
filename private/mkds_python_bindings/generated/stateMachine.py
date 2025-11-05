from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.types import *

state_machine_state_func_t = c_void_p32
state_machine_init_func_t = c_void_p32

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_stateMachine_h_14_9(Structure):
    _fields_ = [
        ('states', POINTER32(state_machine_state_t)),
        ('counter', c_int),
        ('userData', c_void_p32),
        ('nrStates', c_int),
        ('curState', c_int),
        ('nextState', c_int),
        ('gotoNextState', c_int),
    ]

class state_machine_state_t(Structure):
    _fields_ = [
        ('initFunc', state_machine_init_func_t),
        ('stateFunc', state_machine_state_func_t),
    ]

class state_machine_t(Structure):
    _fields_ = [
        ('states', POINTER32(state_machine_state_t)),
        ('counter', u32),
        ('userData', c_void_p32),
        ('nrStates', u16),
        ('curState', u16),
        ('nextState', u16),
        ('gotoNextState', u16),
    ]

