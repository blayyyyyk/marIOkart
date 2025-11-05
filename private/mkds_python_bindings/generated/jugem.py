from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.animationManager import *
from private.mkds_python_bindings.generated.billboardModel import *
from private.mkds_python_bindings.generated.model import *

JG_OBJECT_LAP_COUNT = 0
JG_OBJECT_LAP_FINAL = 1
JG_OBJECT_REVERSE = 2
JG_OBJECT_FLAG = 3
JG_OBJECT_NONE = 4

JG_OBJECT_LAP_COUNT = 0
JG_OBJECT_LAP_FINAL = 1
JG_OBJECT_REVERSE = 2
JG_OBJECT_FLAG = 3
JG_OBJECT_NONE = 4

JG_STATE_IDLE = 0
JG_STATE_REVERSE = 1
JG_STATE_LAP_COUNT = 2
JG_STATE_FLAG = 3
JG_STATE_RESPAWNING = 4
JG_STATE_VANISH = 5
JG_STATE_APPEAR = 6
JG_STATE_COUNT = 7

JG_STATE_IDLE = 0
JG_STATE_REVERSE = 1
JG_STATE_LAP_COUNT = 2
JG_STATE_FLAG = 3
JG_STATE_RESPAWNING = 4
JG_STATE_VANISH = 5
JG_STATE_APPEAR = 6
JG_STATE_COUNT = 7


class struc_235(Structure):
    _fields_ = [
        ('nsbmdName', POINTER32(c_char)),
        ('nsbcaName', POINTER32(c_char)),
        ('nsbmaName', POINTER32(c_char)),
    ]

class struc_236(Structure):
    _fields_ = [
        ('model', POINTER32(model_t)),
        ('nsbcaAnim', POINTER32(anim_manager_t)),
        ('nsbmaAnim', POINTER32(anim_manager_t)),
        ('bbmModel', POINTER32(bbm_model_t)),
    ]
JgState = c_int
JgObject = c_int
