from ctypes import *
from private.mkds_python_bindings.testing.animationManager import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.model import *
from private.mkds_python_bindings.testing.types import *


class bbm_model_t(Structure):
    _fields_ = [
        ('displayList', u32),
        ('displayListLength', u32),
        ('posScale', fx32),
        ('diffAmb', u32),
        ('speEmi', u32),
        ('polygonAttr', u32),
        ('texIdx', u16),
        ('texCount', u16),
        ('texParamList', u32), #POINTER(u32)),
        ('plttParamList', u32), #POINTER(u32)),
        ('model', u32), #POINTER(model_t)),
        ('nsbtpAnim', u32), #POINTER(anim_manager_t)),
    ]
