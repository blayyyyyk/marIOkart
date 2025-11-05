from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.animationManager import *
from private.mkds_python_bindings.generated.billboardModel import *
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.model import *
from private.mkds_python_bindings.generated.model2 import *


class mobj_model_t(Structure):
    _fields_ = [
        ('scale', VecFx32),
        ('bbModel', POINTER32(bbm_model_t)),
        ('model', POINTER32(model_t)),
        ('shadowModel', POINTER32(shadowmodel_t)),
        ('nsbmd', c_void_p32),
        ('nsbcaAnim', POINTER32(anim_manager_t)),
        ('nsbtpAnim', POINTER32(anim_manager_t)),
        ('nsbmaAnim', POINTER32(anim_manager_t)),
        ('nsbtaAnim', POINTER32(anim_manager_t)),
    ]
