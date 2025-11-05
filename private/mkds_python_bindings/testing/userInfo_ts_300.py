from ctypes import *
from private.mkds_python_bindings.testing.types import *
from private.mkds_python_bindings.testing.userInfo_ts_200 import *


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_spi_common_userInfo_ts_300_h_80_16_(Structure):
    _fields_ = [
        ('month', u8),
        ('day', u8),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_spi_common_userInfo_ts_300_h_89_16_(Structure):
    _fields_ = [
        ('str', (u16 * 10)),
        ('length', u8),
        ('rsv', u8),
    ]

class NVRAMConfigComment(Structure):
    _fields_ = [
        ('str', (u16 * 26)),
        ('length', u8),
        ('rsv', u8),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_spi_common_userInfo_ts_300_h_109_16_(Structure):
    _fields_ = [
        ('favoriteColor', u8),
        ('rsv', u8),
        ('birthday', NVRAMConfigDate),
        ('pad', u8),
        ('nickname', NVRAMConfigNickname),
        ('comment', NVRAMConfigComment),
    ]

class NVRAMConfigAlarm(Structure):
    _fields_ = [
        ('hour', u8),
        ('minute', u8),
        ('second', u8),
        ('pad', u8),
        ('enableWeek', u16),
        ('alarmOn', u16),
        ('rsv', u16),
    ]

class NVRAMConfigTpCalibData(Structure):
    _fields_ = [
        ('raw_x1', u16),
        ('raw_y1', u16),
        ('dx1', u8),
        ('dy1', u8),
        ('raw_x2', u16),
        ('raw_y2', u16),
        ('dx2', u8),
        ('dy2', u8),
    ]

class NVRAMConfigOption(Structure):
    _fields_ = [
        ('language', u16),
        ('agbLcd', u16),
        ('detectPullOutCardFlag', u16),
        ('detectPullOutCtrdgFlag', u16),
        ('autoBootFlag', u16),
        ('rsv', u16),
        ('input_favoriteColor', u16),
        ('input_tp', u16),
        ('input_language', u16),
        ('input_rtc', u16),
        ('input_nickname', u16),
        ('timezone', u8),
        ('rtcClockAdjust', u8),
        ('rtcOffset', s64),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_spi_common_userInfo_ts_300_h_174_16_(Structure):
    _fields_ = [
        ('version', u8),
        ('pad', u8),
        ('owner', NVRAMConfigOwnerInfo),
        ('alarm', NVRAMConfigAlarm),
        ('tp', NVRAMConfigTpCalibData),
        ('option', NVRAMConfigOption),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_spi_common_userInfo_ts_300_h_187_16_(Structure):
    _fields_ = [
        ('ncd', NVRAMConfigData),
        ('saveCount', u16),
        ('crc16', u16),
    ]

class NVRAMConfigDataEx(Structure):
    _fields_ = [
        ('version', u8),
        ('language', u8),
        ('valid_language_bitmap', u16),
        ('padding', (u8 * 154)),
    ]

class NVRAMConfigEx(Structure):
    _fields_ = [
        ('ncd', NVRAMConfigData),
        ('saveCount', u16),
        ('crc16', u16),
        ('ncd_ex', NVRAMConfigDataEx),
        ('crc16_ex', u16),
    ]
