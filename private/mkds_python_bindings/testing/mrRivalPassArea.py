from ctypes import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.types import *


class area_mission_rival_pass_area_t(Structure):
    _fields_ = [
        ('index', u16),
        ('size', fx32),
        ('prevNrObjsInside', u8),
        ('passCount', u8),
    ]

class area_mission_rival_pass_area_status_t(Structure):
    _fields_ = [
        ('entries', u32), #POINTER(area_mission_rival_pass_area_t)),
        ('count', u16),
    ]
