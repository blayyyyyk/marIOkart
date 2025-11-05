from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.fx import *
from private.mkds_python_bindings.generated.types import *


class struc_217B4F8(Structure):
    _fields_ = [
        ('field0', fx32),
        ('field4', fx32),
        ('camPos', VecFx32),
        ('camForward', VecFx32),
        ('frustumLeftVec', VecFx32),
        ('frustumRightVec', VecFx32),
        ('frustumTopVec', VecFx32),
        ('frustumBottomVec', VecFx32),
        ('field50', BOOL),
        ('isObjFadeDisabled', BOOL),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_race_mapobj_mapobjRender_h_3_9(Structure):
    _fields_ = [
        ('field0', c_int),
        ('field4', c_int),
        ('camPos', c_int),
        ('camForward', c_int),
        ('frustumLeftVec', c_int),
        ('frustumRightVec', c_int),
        ('frustumTopVec', c_int),
        ('frustumBottomVec', c_int),
        ('field50', c_int),
        ('isObjFadeDisabled', c_int),
    ]
