from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.g2d_Cell_data import *
from private.mkds_python_bindings.generated.mapobjInstance import *
from private.mkds_python_bindings.generated.types import *


class mpicn_mobj_icon_group_t(Structure):
    _fields_ = [
        ('mobjInstanceList', POINTER32(POINTER32(mobj_inst_t))),
        ('cellCount', u32),
        ('mobjInstanceCount', s32),
        ('points', POINTER32(vec2i_t)),
        ('cells', POINTER32(mpicn_mobj_icon_cell_t)),
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

class mpicn_mobj_icon_cell_t(Structure):
    _fields_ = [
        ('position', vec2i_t),
        ('cell', POINTER32(NNSG2dCellData)),
        ('priority', u16),
        ('flipX', u16),
        ('rotation', u16),
        ('field12', u16),
    ]
mpicn_create_func_t = c_void_p32
mpicn_update_func_t = c_void_p32
mpicn_destroy_func_t = c_void_p32
mpicn_render_func_t = c_void_p32

class vec2i_t(Structure):
    _fields_ = [
        ('x', s32),
        ('y', s32),
    ]
