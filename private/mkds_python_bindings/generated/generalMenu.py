from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_scene_generalMenu_h_3_9(Structure):
    _fields_ = [
        ('unknownLoaded', c_int),
        ('selectChoisesLoaded', c_int),
        ('selectReturnLoaded', c_int),
        ('fieldC', c_int),
        ('screenTmpBuf', (c_int * 1024)),
        ('charVramLeft', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_scene_generalMenu_h_13_9(Structure):
    _fields_ = [
        ('loadUnknown', c_int),
        ('unkFont', POINTER32(c_int)),
        ('field8', c_int),
        ('fieldC', c_int),
        ('field10', c_int),
        ('loadSelectChoises', c_int),
        ('selectChoisesFont', POINTER32(c_int)),
        ('loadSelectReturn', c_int),
        ('loadBackground', c_int),
        ('field24', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_scene_generalMenu_h_27_9(Structure):
    _fields_ = [
        ('field0', c_int),
        ('field4', c_int),
        ('field8', c_int),
        ('fieldC', c_int),
        ('screenTmpBuf', (c_int * 1024)),
        ('field810', c_int),
        ('field814', c_int),
        ('field818', c_int),
        ('field81C', c_int),
        ('field820', c_int),
        ('seqArcIndex', c_int),
    ]
