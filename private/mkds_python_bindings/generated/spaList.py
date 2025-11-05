from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.types import *


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_jnlib_spa_spaList_h_9_9(Structure):
    _fields_ = [
        ('head', POINTER32(spa_list_link_t)),
        ('count', c_int),
        ('tail', POINTER32(spa_list_link_t)),
    ]

class spa_list_t(Structure):
    _fields_ = [
        ('head', POINTER32(spa_list_link_t)),
        ('count', u32),
        ('tail', POINTER32(spa_list_link_t)),
    ]

class spa_list_link_t(Structure):
    _fields_ = [
        ('next', POINTER32(spa_list_link_t)),
        ('prev', POINTER32(spa_list_link_t)),
    ]
