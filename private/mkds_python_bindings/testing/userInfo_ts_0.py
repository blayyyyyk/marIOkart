from ctypes import *
from private.mkds_python_bindings.testing.types import *
from private.mkds_python_bindings.testing.userInfo_teg import *
from private.mkds_python_bindings.testing.userInfo_ts_200 import *

NVRAMConfigBloodType = c_int
NVRAMConfigSexCode = c_int


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_spi_common_userInfo_ts_0_h_60_16_(Structure):
    _fields_ = [
        ('name', (u16 * 8)),
        ('length', u16),
        ('padding', u16),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_spi_common_userInfo_ts_0_h_90_16_(Structure):
    _fields_ = [
        ('year', u16),
        ('month', u8),
        ('day', u8),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_spi_common_userInfo_ts_0_h_100_16_(Structure):
    _fields_ = [
        ('nickname', NVRAMConfigNickname),
        ('sex', NVRAMConfigSexCode),
        ('bloodType', NVRAMConfigBloodType),
        ('birthday', NVRAMConfigDate),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_spi_common_userInfo_ts_0_h_112_16_(Structure):
    _fields_ = [
        ('gameCode', (u32 * 8)),
        ('top', u16),
        ('num', u16),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_spi_common_userInfo_ts_0_h_122_16_(Structure):
    _fields_ = [
        ('calib_data', (u16 * 6)),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_spi_common_userInfo_ts_0_h_131_16_(Structure):
    _fields_ = [
        ('version', u8),
        ('timezone', u8),
        ('agbLcd', u16),
        ('rtcOffset', s64),
        ('language', u32),
        ('owner', NVRAMConfigOwnerInfo),
        ('tp', NVRAMConfigTpCData),
        ('bootGameLog', NVRAMConfigBootGameLog),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_spi_common_userInfo_ts_0_h_146_16_(Structure):
    _fields_ = [
        ('ncd', NVRAMConfigData),
        ('saveCount', u16),
        ('crc16', u16),
    ]
