from ctypes import *
from private.mkds_python_bindings.testing.animationManager import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.model import *
from private.mkds_python_bindings.testing.types import *


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_race_courseModel_h_5_9_(Structure):
    _fields_ = [
        ('model', u32), #POINTER(model_t)),
        ('modelV', u32), #POINTER(model_t)),
        ('nsbtpAnim', anim_manager_t),
        ('nsbtaAnim', anim_manager_t),
        ('nsbtaAnimV', anim_manager_t),
        ('mtx', MtxFx43),
        ('modelHasPartialFog', BOOL),
        ('modelFogFlags', s64),
        ('modelVHasPartialFog', BOOL),
        ('modelVFogFlags', u16),
    ]
