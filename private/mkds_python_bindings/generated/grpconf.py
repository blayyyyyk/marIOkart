from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_mapobj_grpconf_h_3_9(Structure):
    _fields_ = [
        ('objectId', c_int),
        ('has3DModel', c_int),
        ('nearClip', c_int),
        ('farClip', c_int),
        ('collisionType', c_int),
        ('width', c_int),
        ('height', c_int),
        ('depth', c_int),
    ]
