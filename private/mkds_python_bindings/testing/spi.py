from ctypes import *
from private.mkds_python_bindings.testing.microphone import *
from private.mkds_python_bindings.testing.types import *

SPI_BAUDRATE_4MHZ = 0
SPI_BAUDRATE_2MHZ = 1
SPI_BAUDRATE_1MHZ = 2
SPI_BAUDRATE_512KHZ = 3

SPI_BAUDRATE_4MHZ = 0
SPI_BAUDRATE_2MHZ = 1
SPI_BAUDRATE_1MHZ = 2
SPI_BAUDRATE_512KHZ = 3

SPI_TRANSMODE_1BYTE = 0
SPI_TRANSMODE_CONTINUOUS = 1

SPI_TRANSMODE_1BYTE = 0
SPI_TRANSMODE_CONTINUOUS = 1

SPI_COMMPARTNER_PMIC = 0
SPI_COMMPARTNER_EEPROM = 1

SPI_COMMPARTNER_PMIC = 0
SPI_COMMPARTNER_EEPROM = 1

SPIBaudRate = c_int
SPICommPartner = c_int
SPITransMode = c_int


class input_tpmic_tp_t(Structure):
    _fields_ = [
        ('tpX', u16),
        ('tpY', u16),
        ('tpValid', u16),
        ('field6', u16),
    ]

class input_tpmic_t(Structure):
    _fields_ = [
        ('curTp', input_tpmic_tp_t),
        ('prevTp', input_tpmic_tp_t),
        ('tpReleaseFrameCounter', u16),
        ('mic', u8),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_input_spi_h_20_9_(Structure):
    _fields_ = [
        ('tpSampleBuf', (c_int * 5)),
        ('curTp', c_int),
        ('tpAutoSamplingEnabled', BOOL),
        ('micAutoSamplingEnabled', BOOL),
        ('micAutoSamplingPaused', BOOL),
        ('micInputDetected', BOOL),
        ('gap34', (u8 * 8)),
        ('micData', u32), #POINTER(mic_t)),
    ]
