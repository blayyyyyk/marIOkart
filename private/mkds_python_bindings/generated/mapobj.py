from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.item import *
from private.mkds_python_bindings.generated.mapobjConfig import *
from private.mkds_python_bindings.generated.mapobjInstance import *
from private.mkds_python_bindings.generated.mapobjLogicPart import *
from private.mkds_python_bindings.generated.mapobjRenderPart import *
from private.mkds_python_bindings.generated.types import *


class mobj_table_entry_t(Structure):
    _fields_ = [
        ('id', u32),
        ('def', POINTER32(mobj_def_t)),
        ('arg', c_void_p32),
    ]

class mobj_state_t(Structure):
    _fields_ = [
        ('mobjInstanceList', POINTER32(POINTER32(mobj_inst_t))),
        ('mobjInstanceCount', u16),
        ('renderPartList', (POINTER32(mobj_render_part_t) * 24)),
        ('renderPart3dList', (POINTER32(mobj_render_part_t) * 24)),
        ('renderPart2dList', (POINTER32(mobj_render_part_t) * 24)),
        ('renderPartCount', u16),
        ('renderPart3dCount', u16),
        ('renderPart2dCount', u16),
        ('field130', u32),
        ('logicPartList', (POINTER32(mobj_logic_part_t) * 16)),
        ('logicPartCount', u16),
        ('field178', u32),
        ('hasKoopaBlock', BOOL),
        ('hasRotatingCylinder', BOOL),
        ('hasBridge', BOOL),
        ('hasWall', BOOL),
        ('pseudoItem', it_item_inst_t),
        ('logicUpdateEnabled', BOOL),
    ]

class mobj_def_t(Structure):
    _fields_ = [
        ('instanceCount', u16),
        ('instanceSize', u32),
        ('instInitFunc', mobj_inst_init_func_t),
        ('configSetupFunc', mobj_config_setup_func_t),
        ('renderPartSetupFuncs', (mobj_render_part_setup_func_t * 3)),
        ('logicPartSetupFunc', mobj_logic_part_setup_func_t),
        ('config', POINTER32(mobj_config_t)),
        ('renderParts', (POINTER32(mobj_render_part_t) * 3)),
        ('logicPart', POINTER32(mobj_logic_part_t)),
    ]

class idk_struct2_t(Structure):
    _fields_ = [
        ('value', fx32),
        ('velocity', fx32),
        ('min', fx32),
        ('max', fx32),
    ]

class idk_struct_t(Structure):
    _fields_ = [
        ('value', fx32),
        ('velocity', fx32),
        ('reverse', BOOL),
    ]
mobj_logic_part_setup_func_t = c_void_p32
mobj_config_setup_func_t = c_void_p32
mobj_render_part_setup_func_t = c_void_p32
mobj_inst_init_func_t = c_void_p32
