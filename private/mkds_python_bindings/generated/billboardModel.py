from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.animationManager import *
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.model import *
from private.mkds_python_bindings.generated.types import *


class bbm_model_t(Structure):
    _fields_ = [
        ('displayList', c_void_p32),
        ('displayListLength', u32),
        ('posScale', fx32),
        ('diffAmb', u32),
        ('speEmi', u32),
        ('polygonAttr', u32),
        ('texIdx', u16),
        ('texCount', u16),
        ('texParamList', POINTER32(u32)),
        ('plttParamList', POINTER32(u32)),
        ('model', POINTER32(model_t)),
        ('nsbtpAnim', POINTER32(anim_manager_t)),
    ]
