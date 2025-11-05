from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_jnlib_ui_jnLyt_h_6_9(Structure):
    _fields_ = [
        ('charCanvas', c_int),
        ('textCanvas', c_int),
        ('charData', c_void_p32),
        ('charDataLength', c_int),
        ('charDataTileOffset', c_int),
        ('width', c_int),
        ('height', c_int),
        ('cellData', POINTER32(c_int)),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_jnlib_ui_jnLyt_h_18_9(Structure):
    _fields_ = [
        ('visible', c_int),
        ('offsetX', c_int),
        ('offsetY', c_int),
        ('usePosition', c_int),
        ('positionX', c_int),
        ('positionY', c_int),
        ('useMtx', c_int),
        ('baseMtx', c_int),
        ('affineMtx', c_int),
        ('useDoubleAffine', c_int),
        ('subElement', c_int),
        ('label', POINTER32(jnui_label_t)),
    ]
