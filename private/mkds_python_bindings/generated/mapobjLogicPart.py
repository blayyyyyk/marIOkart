from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.mapobjInstance import *
from private.mkds_python_bindings.generated.types import *


class mobj_logic_part_t(Structure):
    _fields_ = [
        ('mobjInstanceList', POINTER32(POINTER32(mobj_inst_t))),
        ('mobjInstanceCount', u16),
        ('globalInitFunc', mobj_logic_part_global_init_func_t),
        ('globalPreUpdateFunc', mobj_logic_part_global_pre_update_func_t),
        ('instanceUpdateFunc', mobj_logic_part_instance_update_func_t),
        ('globalPostUpdateFunc', mobj_logic_part_global_post_update_func_t),
        ('type', u32),
        ('thunderFunc', mobj_logic_part_thunder_func_t),
        ('thunderObjResp', u32),
        ('thisPointer', POINTER32(POINTER32(mobj_logic_part_t))),
    ]
mobj_logic_part_global_init_func_t = c_void_p32
mobj_logic_part_global_pre_update_func_t = c_void_p32
mobj_logic_part_global_post_update_func_t = c_void_p32
mobj_logic_part_thunder_func_t = c_void_p32
mobj_logic_part_instance_update_func_t = c_void_p32
