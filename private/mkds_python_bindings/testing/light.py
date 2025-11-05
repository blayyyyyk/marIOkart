from ctypes import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.gxcommon import *
from private.mkds_python_bindings.testing.types import *


class light_t(Structure):
    _fields_ = [
        ('color', GXRgb),
        ('r', s16),
        ('g', s16),
        ('b', s16),
        ('rDelta', s16),
        ('gDelta', s16),
        ('bDelta', s16),
        ('lightMask', u16),
        ('progress', fx16),
    ]
