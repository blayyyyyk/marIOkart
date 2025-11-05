from ctypes import *
from private.mkds_python_bindings.testing.nnsfnd import *
from private.mkds_python_bindings.testing.rand import *
from private.mkds_python_bindings.testing.sound3 import *
from private.mkds_python_bindings.testing.types import *

MKDS_LANGUAGE_JA = 0
MKDS_LANGUAGE_US = 1
MKDS_LANGUAGE_FR = 2
MKDS_LANGUAGE_GE = 3
MKDS_LANGUAGE_IT = 4
MKDS_LANGUAGE_ES = 5
MKDS_LANGUAGE_US2 = 6

MKDS_LANGUAGE_JA = 0
MKDS_LANGUAGE_US = 1
MKDS_LANGUAGE_FR = 2
MKDS_LANGUAGE_GE = 3
MKDS_LANGUAGE_IT = 4
MKDS_LANGUAGE_ES = 5
MKDS_LANGUAGE_US2 = 6

MKDS_OVERLAYRELATEDNUM_0 = 0
MKDS_OVERLAYRELATEDNUM_1 = 1
MKDS_OVERLAYRELATEDNUM_MULTIPLAYER = 2
MKDS_OVERLAYRELATEDNUM_WIFI = 3
MKDS_OVERLAYRELATEDNUM_WIFI_UTIL = 4

MKDS_OVERLAYRELATEDNUM_0 = 0
MKDS_OVERLAYRELATEDNUM_1 = 1
MKDS_OVERLAYRELATEDNUM_MULTIPLAYER = 2
MKDS_OVERLAYRELATEDNUM_WIFI = 3
MKDS_OVERLAYRELATEDNUM_WIFI_UTIL = 4

SYSDAT_BACKLIGHT_STATE_ON = 0
SYSDAT_BACKLIGHT_STATE_OFF = 1

SYSDAT_BACKLIGHT_STATE_ON = 0
SYSDAT_BACKLIGHT_STATE_OFF = 1

MKDSLanguage = c_int
MKDSOverlayRelatedNum = c_int
SysdatBacklightState = c_int


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_systemData_h_31_9_(Structure):
    _fields_ = [
        ('field0', u32),
        ('language', MKDSLanguage),
        ('seqHandle', seq_handle_t),
        ('isSeqPlaying', BOOL),
        ('isSeqLoaded', BOOL),
        ('isMBChild', BOOL),
        ('useG3dFastDma', BOOL),
        ('overlayRelatedNum', MKDSOverlayRelatedNum),
        ('dtcmHeapHandle', NNSFndHeapHandle),
        ('field28', s16),
        ('nickName', u16),
        ('nickNameLength', u16),
        ('favoriteColor', u8),
        ('activatedRaceMenuOption', c_int),
        ('backLightTop', c_int),
        ('backLightBottom', c_int),
        ('field50', u16),
        ('field54', u32),
        ('field58', u32),
        ('field5C', u32),
        ('random', MATHRandContext32),
        ('field78', u32),
    ]
