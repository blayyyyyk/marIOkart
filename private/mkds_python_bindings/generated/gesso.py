from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.driver import *
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.item import *
from private.mkds_python_bindings.generated.types import *


class struc_373(Structure):
    _fields_ = [
        ('field0', s16),
        ('field2', s16),
        ('field4', s16),
        ('field6', s16),
        ('field8', u16),
        ('fieldA', u16),
    ]

class gesso_t(Structure):
    _fields_ = [
        ('item', it_item_inst_t),
        ('driver', POINTER32(driver_t)),
        ('driverSplashCount', (u8 * 8)),
        ('field138', c_int),
        ('field13C', VecFx32),
        ('field148', VecFx32),
        ('field154', VecFx32),
        ('field160', BOOL),
        ('field164', c_int),
        ('field168', c_int),
        ('field16C', c_int),
        ('field170', c_int),
        ('field174', VecFx32),
        ('field180', VecFx32),
        ('field18C', c_int),
        ('gap190', u32),
        ('field194', c_int),
        ('visible', BOOL),
        ('field19C', fx32),
        ('gap1A0', (u8 * 20)),
    ]
