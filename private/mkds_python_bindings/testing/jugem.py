from ctypes import *
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

JgObject = c_int
JgState = c_int


class struc_235(Structure):
    _fields_ = [
        ('nsbmdName', u32), #POINTER(c_char)),
        ('nsbcaName', u32), #POINTER(c_char)),
        ('nsbmaName', u32), #POINTER(c_char)),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_race_jugem_h_32_9_(Structure):
    _fields_ = [
        ('model', u32), #POINTER(c_int)),
        ('nsbcaAnim', u32), #POINTER(c_int)),
        ('nsbmaAnim', u32), #POINTER(c_int)),
        ('bbmModel', u32), #POINTER(c_int)),
    ]
