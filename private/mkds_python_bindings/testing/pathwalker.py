from ctypes import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.nkm import *
from private.mkds_python_bindings.testing.types import *

pw_progress_t = c_int
pw_progress_t = c_int
pw_progress_t = c_int
pw_progress_t = c_int
pw_progress_t = c_int
pw_progress_t = c_int
pw_progress_t = c_int
pw_progress_t = c_int
pw_progress_t = c_int
pw_progress_t = c_int
pw_progress_t = c_int
pw_progress_t = c_int
pw_progress_t = c_int
pw_progress_t = c_int
pw_progress_t = c_int
pw_progress_t = c_int
pw_progress_t = c_int
pw_progress_t = c_int
pw_progress_t = c_int


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

class pw_simple_pathwalker_t(Structure):
    _fields_ = [
        ('pathPart', pw_path_part_t),
        ('speed', fx32),
        ('partSpeed', pw_progress_t),
        ('progress', pw_progress_t),
    ]

class pw_path_t(Structure):
    _fields_ = [
        ('parts', u32), #POINTER(pw_path_part_t)),
        ('partCount', u32),
        ('loop', BOOL),
    ]

class pw_pathwalker_t(Structure):
    _fields_ = [
        ('path', u32), #POINTER(pw_path_t)),
        ('speed', fx32),
        ('pathId', u16),
        ('partIdx', u32),
        ('partSpeed', pw_progress_t),
        ('partProgress', pw_progress_t),
        ('isForwards', BOOL),
        ('prevPoit', u32), #POINTER(nkm_poit_entry_t)),
        ('curPoit', u32), #POINTER(nkm_poit_entry_t)),
    ]
