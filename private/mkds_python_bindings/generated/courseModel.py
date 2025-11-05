from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.animationManager import *
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.model import *
from private.mkds_python_bindings.generated.types import *


class crsmdl_t(Structure):
    _fields_ = [
        ('model', POINTER32(model_t)),
        ('modelV', POINTER32(model_t)),
        ('nsbtpAnim', anim_manager_t),
        ('nsbtaAnim', anim_manager_t),
        ('nsbtaAnimV', anim_manager_t),
        ('mtx', MtxFx43),
        ('modelHasPartialFog', BOOL),
        ('modelFogFlags', s64),
        ('modelVHasPartialFog', BOOL),
        ('modelVFogFlags', u16),
    ]
