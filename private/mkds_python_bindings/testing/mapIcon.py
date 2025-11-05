from ctypes import *
from private.mkds_python_bindings.testing.mapobjInstance import *
from private.mkds_python_bindings.testing.types import *

mpicn_create_func_t = u32
mpicn_destroy_func_t = u32
mpicn_render_func_t = u32
mpicn_update_func_t = u32


class vec2i_t(Structure):
    _fields_ = [
        ('x', s32),
        ('y', s32),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_race_race2d_map2d_mapIcon_h_13_9_(Structure):
    _fields_ = [
        ('position', vec2i_t),
        ('cell', u32), #POINTER(c_int)),
        ('priority', u16),
        ('flipX', u16),
        ('rotation', u16),
        ('field12', u16),
    ]

class mpicn_mobj_icon_group_t(Structure):
    _fields_ = [
        ('mobjInstanceList', u32), #POINTER(POINTER(mobj_inst_t))),
        ('cellCount', u32),
        ('mobjInstanceCount', s32),
        ('points', u32), #POINTER(vec2i_t)),
        ('cells', u32), #POINTER(mpicn_mobj_icon_cell_t)),
    ]

class mpicn_def_t(Structure):
    _fields_ = [
        ('createFunc', mpicn_create_func_t),
        ('destroyFunc', mpicn_destroy_func_t),
        ('updateFunc', mpicn_update_func_t),
        ('renderFunc', mpicn_render_func_t),
    ]

class mpicn_icon_data_t(Structure):
    _fields_ = [
        ('minPriority', u16),
        ('field4', u32),
        ('field8', u32),
        ('groupCount', u32),
        ('destroyFuncs', (mpicn_destroy_func_t * 49)),
        ('updateFuncs', (mpicn_update_func_t * 49)),
        ('renderFuncs', (mpicn_render_func_t * 49)),
    ]
