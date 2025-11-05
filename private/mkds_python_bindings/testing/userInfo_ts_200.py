from ctypes import *
from private.mkds_python_bindings.testing.types import *
from private.mkds_python_bindings.testing.userInfo_teg import *
from private.mkds_python_bindings.testing.userInfo_ts_300 import *


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_spi_common_userInfo_ts_200_h_61_16_(Structure):
    _fields_ = [
        ('month', u8),
        ('day', u8),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_spi_common_userInfo_ts_200_h_70_16_(Structure):
    _fields_ = [
        ('str', (u16 * 10)),
        ('length', u8),
        ('rsv', u8),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_spi_common_userInfo_ts_200_h_90_16_(Structure):
    _fields_ = [
        ('favoriteColor', u8),
        ('rsv', u8),
        ('birthday', NVRAMConfigDate),
        ('pad', u8),
        ('nickname', NVRAMConfigNickname),
        ('comment', NVRAMConfigComment),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_spi_common_userInfo_ts_200_h_155_16_(Structure):
    _fields_ = [
        ('version', u8),
        ('pad', u8),
        ('owner', NVRAMConfigOwnerInfo),
        ('alarm', NVRAMConfigAlarm),
        ('tp', NVRAMConfigTpCalibData),
        ('option', NVRAMConfigOption),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_spi_common_userInfo_ts_200_h_168_16_(Structure):
    _fields_ = [
        ('ncd', NVRAMConfigData),
        ('saveCount', u16),
        ('crc16', u16),
    ]
