from ctypes import *
from private.mkds_python_bindings.testing.types import *
from private.mkds_python_bindings.testing.userInfo_ts_0 import *
from private.mkds_python_bindings.testing.userInfo_ts_200 import *

NVRAM_CONFIG_SEX_MALE = 0
NVRAM_CONFIG_SEX_FEMALE = 1
NVRAM_CONFIG_SEX_CODE_MAX = 2

NVRAM_CONFIG_SEX_MALE = 0
NVRAM_CONFIG_SEX_FEMALE = 1
NVRAM_CONFIG_SEX_CODE_MAX = 2

NVRAM_CONFIG_BLOOD_A = 0
NVRAM_CONFIG_BLOOD_B = 1
NVRAM_CONFIG_BLOOD_AB = 2
NVRAM_CONFIG_BLOOD_O = 3
NVRAM_CONFIG_BLOOD_TYPE_MAX = 4

NVRAM_CONFIG_BLOOD_A = 0
NVRAM_CONFIG_BLOOD_B = 1
NVRAM_CONFIG_BLOOD_AB = 2
NVRAM_CONFIG_BLOOD_O = 3
NVRAM_CONFIG_BLOOD_TYPE_MAX = 4

NVRAMConfigBloodType = c_int
NVRAMConfigSexCode = c_int


class NVRAMConfigNickname(Structure):
    _fields_ = [
        ('name', (u16 * 8)),
        ('length', u16),
        ('padding', u16),
    ]

class NVRAMConfigDate(Structure):
    _fields_ = [
        ('year', u16),
        ('month', u8),
        ('day', u8),
    ]

class NVRAMConfigOwnerInfo(Structure):
    _fields_ = [
        ('nickname', NVRAMConfigNickname),
        ('sex', NVRAMConfigSexCode),
        ('bloodType', NVRAMConfigBloodType),
        ('birthday', NVRAMConfigDate),
    ]

class NVRAMConfigBootGameLog(Structure):
    _fields_ = [
        ('gameCode', (u32 * 8)),
        ('top', u16),
        ('num', u16),
    ]

class NVRAMConfigTpCData(Structure):
    _fields_ = [
        ('calib_data', (u16 * 6)),
    ]

class NVRAMConfigData(Structure):
    _fields_ = [
        ('version', u8),
        ('timezone', u8),
        ('agbLcd', u16),
        ('rtcOffset', u32),
        ('language', u32),
        ('owner', NVRAMConfigOwnerInfo),
        ('tp', NVRAMConfigTpCData),
        ('bootGameLog', NVRAMConfigBootGameLog),
    ]

class NVRAMConfig(Structure):
    _fields_ = [
        ('ncd', NVRAMConfigData),
        ('saveCount', u16),
        ('crc16', u16),
    ]
