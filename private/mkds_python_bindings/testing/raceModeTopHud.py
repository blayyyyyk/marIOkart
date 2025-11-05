from ctypes import *
from private.mkds_python_bindings.testing.types import *


class r2d_race_mode_top_hud_state_t(Structure):
    _fields_ = [
        ('mrTargetValue', c_int),
        ('ghostAvailable', BOOL),
    ]

class union__anonymous_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_race_race2d_raceModeTopHud_h_5_5_(Union):
    _fields_ = [
        ('place', c_int),
        ('mrCurrentValue', c_int),
    ]

class struct__anonymous_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_race_race2d_raceModeTopHud_h_7_9_(Structure):
    _fields_ = [
        ('mgBalloonInflateFrame', u16),
        ('mgInventoryBalloonCount', u16),
    ]
