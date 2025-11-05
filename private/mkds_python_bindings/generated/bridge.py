from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.dynamicCollision import *
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.mapobjRenderPart import *
from private.mkds_python_bindings.generated.types import *


class bridge_renderpart_t(Structure):
    _fields_ = [
        ('renderPart', mobj_render_part_t),
        ('animLength', fx32),
    ]

class bridge_t(Structure):
    _fields_ = [
        ('dcolMObj', dcol_inst_t),
        ('field144', u16),
        ('rotSpeed', u16),
        ('angle', u16),
        ('field14A', u16),
        ('field14C', u16),
        ('field14E', u16),
        ('waitCounter', u16),
        ('animProgress', fx32),
    ]
