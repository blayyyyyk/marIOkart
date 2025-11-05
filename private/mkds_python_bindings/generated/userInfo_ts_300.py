from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.userInfo_teg import *


class NVRAMConfigComment(Structure):
    _fields_ = [
        ('str', (c_int * 26)),
        ('length', c_int),
        ('rsv', c_int),
    ]

class NVRAMConfigAlarm(Structure):
    _fields_ = [
        ('hour', c_int),
        ('minute', c_int),
        ('second', c_int),
        ('pad', c_int),
        ('enableWeek', c_int),
        ('alarmOn', c_int),
        ('rsv', c_int),
    ]

class NVRAMConfigTpCalibData(Structure):
    _fields_ = [
        ('raw_x1', c_int),
        ('raw_y1', c_int),
        ('dx1', c_int),
        ('dy1', c_int),
        ('raw_x2', c_int),
        ('raw_y2', c_int),
        ('dx2', c_int),
        ('dy2', c_int),
    ]

class NVRAMConfigOption(Structure):
    _fields_ = [
        ('language', c_int),
        ('agbLcd', c_int),
        ('detectPullOutCardFlag', c_int),
        ('detectPullOutCtrdgFlag', c_int),
        ('autoBootFlag', c_int),
        ('rsv', c_int),
        ('input_favoriteColor', c_int),
        ('input_tp', c_int),
        ('input_language', c_int),
        ('input_rtc', c_int),
        ('input_nickname', c_int),
        ('timezone', c_int),
        ('rtcClockAdjust', c_int),
        ('rtcOffset', c_int),
    ]

class NVRAMConfigEx(Structure):
    _fields_ = [
        ('ncd', NVRAMConfigData),
        ('saveCount', c_int),
        ('crc16', c_int),
        ('ncd_ex', NVRAMConfigDataEx),
        ('crc16_ex', c_int),
    ]

class NVRAMConfigDataEx(Structure):
    _fields_ = [
        ('version', c_int),
        ('language', c_int),
        ('valid_language_bitmap', c_int),
        ('padding', (c_int * 245)),
    ]
