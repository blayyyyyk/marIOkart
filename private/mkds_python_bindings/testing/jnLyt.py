from ctypes import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.types import *


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_jnlib_ui_jnLyt_h_6_9_(Structure):
    _fields_ = [
        ('charCanvas', c_int),
        ('textCanvas', c_int),
        ('charData', u32),
        ('charDataLength', u32),
        ('charDataTileOffset', u32),
        ('width', u32),
        ('height', u32),
        ('cellData', u32), #POINTER(c_int)),
    ]

class jnui_layout_element_t(Structure):
    _fields_ = [
        ('visible', BOOL),
        ('offsetX', s16),
        ('offsetY', s16),
        ('usePosition', BOOL),
        ('positionX', s16),
        ('positionY', s16),
        ('useMtx', BOOL),
        ('baseMtx', MtxFx22),
        ('affineMtx', MtxFx22),
        ('useDoubleAffine', BOOL),
        ('subElement', s32),
        ('label', u32), #POINTER(jnui_label_t)),
    ]
