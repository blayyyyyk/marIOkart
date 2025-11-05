from ctypes import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.types import *


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
