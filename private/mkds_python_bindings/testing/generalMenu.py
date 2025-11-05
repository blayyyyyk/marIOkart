from ctypes import *
from private.mkds_python_bindings.testing.types import *


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_scene_generalMenu_h_3_9_(Structure):
    _fields_ = [
        ('unknownLoaded', BOOL),
        ('selectChoisesLoaded', BOOL),
        ('selectReturnLoaded', BOOL),
        ('fieldC', u32),
        ('screenTmpBuf', (c_int * 1024)),
        ('charVramLeft', u32),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_scene_generalMenu_h_13_9_(Structure):
    _fields_ = [
        ('loadUnknown', BOOL),
        ('unkFont', u32), #POINTER(c_int)),
        ('field8', u32),
        ('fieldC', u32),
        ('field10', u32),
        ('loadSelectChoises', BOOL),
        ('selectChoisesFont', u32), #POINTER(c_int)),
        ('loadSelectReturn', BOOL),
        ('loadBackground', BOOL),
        ('field24', u32),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_scene_generalMenu_h_27_9_(Structure):
    _fields_ = [
        ('field0', u32),
        ('field4', u32),
        ('field8', u32),
        ('fieldC', u32),
        ('screenTmpBuf', (c_int * 1024)),
        ('field810', u32),
        ('field814', u32),
        ('field818', u32),
        ('field81C', u32),
        ('field820', BOOL),
        ('seqArcIndex', u32),
    ]
