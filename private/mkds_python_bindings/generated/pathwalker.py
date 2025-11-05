from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.nkm import *
from private.mkds_python_bindings.generated.types import *


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_pathwalker_h_11_9(Structure):
    _fields_ = [
        ('p0', c_int),
        ('p1', c_int),
        ('p2', c_int),
        ('p3', c_int),
        ('length', c_int),
        ('oneDivLength', pw_progress_t),
        ('hermLength', c_int),
        ('oneDivHermLength', pw_progress_t),
        ('linLength', c_int),
        ('oneDivLinLength', pw_progress_t),
        ('field48', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_pathwalker_h_26_9(Structure):
    _fields_ = [
        ('pathPart', pw_path_part_t),
        ('speed', c_int),
        ('partSpeed', pw_progress_t),
        ('progress', pw_progress_t),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_pathwalker_h_34_9(Structure):
    _fields_ = [
        ('parts', POINTER32(pw_path_part_t)),
        ('partCount', c_int),
        ('loop', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_pathwalker_h_41_9(Structure):
    _fields_ = [
        ('path', POINTER32(pw_path_t)),
        ('speed', c_int),
        ('pathId', c_int),
        ('partIdx', c_int),
        ('partSpeed', pw_progress_t),
        ('partProgress', pw_progress_t),
        ('isForwards', c_int),
        ('prevPoit', POINTER32(nkm_poit_entry_t)),
        ('curPoit', POINTER32(nkm_poit_entry_t)),
    ]

class pw_simple_pathwalker_t(Structure):
    _fields_ = [
        ('pathPart', pw_path_part_t),
        ('speed', fx32),
        ('partSpeed', pw_progress_t),
        ('progress', pw_progress_t),
    ]

class pw_pathwalker_t(Structure):
    _fields_ = [
        ('path', POINTER32(pw_path_t)),
        ('speed', fx32),
        ('pathId', u16),
        ('partIdx', u32),
        ('partSpeed', pw_progress_t),
        ('partProgress', pw_progress_t),
        ('isForwards', BOOL),
        ('prevPoit', POINTER32(nkm_poit_entry_t)),
        ('curPoit', POINTER32(nkm_poit_entry_t)),
    ]

class pw_path_t(Structure):
    _fields_ = [
        ('parts', POINTER32(pw_path_part_t)),
        ('partCount', u32),
        ('loop', BOOL),
    ]

class pw_path_part_t(Structure):
    _fields_ = [
        ('p0', VecFx32),
        ('p1', VecFx32),
        ('p2', VecFx32),
        ('p3', VecFx32),
        ('length', fx32),
        ('oneDivLength', pw_progress_t),
        ('hermLength', fx32),
        ('oneDivHermLength', pw_progress_t),
        ('linLength', fx32),
        ('oneDivLinLength', pw_progress_t),
        ('field48', VecFx32),
    ]
pw_progress_t = c_int
