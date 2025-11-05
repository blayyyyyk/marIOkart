from ctypes import *
from private.mkds_python_bindings.testing.types import *

SPI_DEVICE_TYPE_TP = 0
SPI_DEVICE_TYPE_NVRAM = 1
SPI_DEVICE_TYPE_MIC = 2
SPI_DEVICE_TYPE_PM = 3
SPI_DEVICE_TYPE_ARM7 = 4
SPI_DEVICE_TYPE_MAX = 5

SPI_DEVICE_TYPE_TP = 0
SPI_DEVICE_TYPE_NVRAM = 1
SPI_DEVICE_TYPE_MIC = 2
SPI_DEVICE_TYPE_PM = 3
SPI_DEVICE_TYPE_ARM7 = 4
SPI_DEVICE_TYPE_MAX = 5

SPI_TP_TOUCH_OFF = 0
SPI_TP_TOUCH_ON = 1

SPI_TP_TOUCH_OFF = 0
SPI_TP_TOUCH_ON = 1

SPI_TP_VALIDITY_VALID = 0
SPI_TP_VALIDITY_INVALID_X = 1
SPI_TP_VALIDITY_INVALID_Y = 2
SPI_TP_VALIDITY_INVALID_XY = 3

SPI_TP_VALIDITY_VALID = 0
SPI_TP_VALIDITY_INVALID_X = 1
SPI_TP_VALIDITY_INVALID_Y = 2
SPI_TP_VALIDITY_INVALID_XY = 3

SPIDeviceType = c_int
SPITpTouch = c_int
SPITpValidity = c_int


class RTCRawDate(Structure):
    _fields_ = [
        ('year', u32),
        ('month', u32),
        ('dummy0', u32),
        ('day', u32),
        ('dummy1', u32),
        ('week', u32),
        ('dummy2', u32),
    ]

class RTCRawTime(Structure):
    _fields_ = [
        ('hour', u32),
        ('afternoon', u32),
        ('dummy0', u32),
        ('minute', u32),
        ('dummy1', u32),
        ('second', u32),
        ('dummy2', u32),
    ]

class RTCRawStatus1(Structure):
    _fields_ = [
        ('reset', u16),
        ('format', u16),
        ('dummy0', u16),
        ('intr1', u16),
        ('intr2', u16),
        ('bld', u16),
        ('poc', u16),
        ('dummy1', u16),
    ]

class RTCRawStatus2(Structure):
    _fields_ = [
        ('intr_mode', u16),
        ('dummy0', u16),
        ('intr2_mode', u16),
        ('test', u16),
        ('dummy1', u16),
    ]

class RTCRawAlarm(Structure):
    _fields_ = [
        ('week', u32),
        ('dummy0', u32),
        ('we', u32),
        ('hour', u32),
        ('afternoon', u32),
        ('he', u32),
        ('minute', u32),
        ('me', u32),
        ('dummy2', u32),
    ]

class RTCRawPulse(Structure):
    _fields_ = [
        ('pulse', u32),
        ('dummy', u32),
    ]

class RTCRawAdjust(Structure):
    _fields_ = [
        ('adjust', u32),
        ('dummy', u32),
    ]

class RTCRawFree(Structure):
    _fields_ = [
        ('free', u32),
        ('dummy', u32),
    ]

class RTCRawData(Union):
    _fields_ = [
        ('t', struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_rtc_common_type_h_167_5_),
        ('a', struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_rtc_common_type_h_174_5_),
        ('words', (u32 * 2)),
        ('halfs', (u16 * 4)),
        ('bytes', (u8 * 8)),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_rtc_common_type_h_167_5_(Structure):
    _fields_ = [
        ('date', RTCRawDate),
        ('time', RTCRawTime),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_rtc_common_type_h_174_5_(Structure):
    _fields_ = [
        ('status1', RTCRawStatus1),
        ('status2', RTCRawStatus2),
    ]

class union__anonymous_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_rtc_common_type_h_178_9_(Union):
    _fields_ = [
        ('pulse', RTCRawPulse),
        ('alarm', RTCRawAlarm),
        ('adjust', RTCRawAdjust),
        ('free', RTCRawFree),
    ]

class SPITpData(Union):
    _fields_ = [
        ('e', struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_spi_common_type_h_236_5_),
        ('raw', u32),
        ('bytes', (u8 * 4)),
        ('halfs', (u16 * 2)),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_spi_common_type_h_236_5_(Structure):
    _fields_ = [
        ('x', u32),
        ('y', u32),
        ('touch', u32),
        ('validity', u32),
        ('dummy', u32),
    ]
