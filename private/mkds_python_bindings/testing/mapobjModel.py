from ctypes import *
from private.mkds_python_bindings.testing.animationManager import *
from private.mkds_python_bindings.testing.billboardModel import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.model import *
from private.mkds_python_bindings.testing.model2 import *


class mobj_model_t(Structure):
    _fields_ = [
        ('scale', VecFx32),
        ('bbModel', u32), #POINTER(bbm_model_t)),
        ('model', u32), #POINTER(model_t)),
        ('shadowModel', u32), #POINTER(shadowmodel_t)),
        ('nsbmd', u32),
        ('nsbcaAnim', u32), #POINTER(anim_manager_t)),
        ('nsbtpAnim', u32), #POINTER(anim_manager_t)),
        ('nsbmaAnim', u32), #POINTER(anim_manager_t)),
        ('nsbtaAnim', u32), #POINTER(anim_manager_t)),
    ]
