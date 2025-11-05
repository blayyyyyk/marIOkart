from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_mapData_mrRivalPassArea_h_7_9(Structure):
    _fields_ = [
        ('index', c_int),
        ('size', c_int),
        ('prevNrObjsInside', c_int),
        ('passCount', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_mapData_mrRivalPassArea_h_15_9(Structure):
    _fields_ = [
        ('entries', POINTER32(area_mission_rival_pass_area_t)),
        ('count', c_int),
    ]
