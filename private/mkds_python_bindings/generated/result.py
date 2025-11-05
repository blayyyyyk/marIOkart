from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.oam import *
from private.mkds_python_bindings.generated.raceConfig import *


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_scene_result_result_h_6_9(Structure):
    _fields_ = [
        ('mainOamBuf', oam_buf_t),
        ('subOamBuf', oam_buf_t),
        ('raceMode', RaceMode),
    ]
