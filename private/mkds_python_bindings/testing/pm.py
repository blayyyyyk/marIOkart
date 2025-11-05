from ctypes import *
from private.mkds_python_bindings.testing.types import *

PM_LCD_TOP = 0
PM_LCD_BOTTOM = 1
PM_LCD_ALL = 2

PM_LCD_TOP = 0
PM_LCD_BOTTOM = 1
PM_LCD_ALL = 2

PM_BACKLIGHT_OFF = 0
PM_BACKLIGHT_ON = 1

PM_BACKLIGHT_OFF = 0
PM_BACKLIGHT_ON = 1

PM_BATTERY_HIGH = 0
PM_BATTERY_LOW = 1

PM_BATTERY_HIGH = 0
PM_BATTERY_LOW = 1

PM_AMP_OFF = 0
PM_AMP_ON = 1

PM_AMP_OFF = 0
PM_AMP_ON = 1

PM_AMPGAIN_20 = 0
PM_AMPGAIN_40 = 1
PM_AMPGAIN_80 = 2
PM_AMPGAIN_160 = 3

PM_AMPGAIN_20 = 0
PM_AMPGAIN_40 = 1
PM_AMPGAIN_80 = 2
PM_AMPGAIN_160 = 3

PM_LCD_POWER_OFF = 0
PM_LCD_POWER_ON = 1

PM_LCD_POWER_OFF = 0
PM_LCD_POWER_ON = 1

PM_SOUND_POWER_OFF = 0
PM_SOUND_POWER_ON = 1

PM_SOUND_POWER_OFF = 0
PM_SOUND_POWER_ON = 1

PM_SOUND_VOLUME_OFF = 0
PM_SOUND_VOLUME_ON = 1

PM_SOUND_VOLUME_OFF = 0
PM_SOUND_VOLUME_ON = 1

PMAmpGain = c_int
PMAmpSwitch = c_int
PMBackLightSwitch = c_int
PMBattery = c_int
PMCallback = u32
PMLCDPower = c_int
PMLCDTarget = c_int
PMSleepCallback = u32
PMSoundPowerSwitch = c_int
PMSoundVolumeSwitch = c_int


class PMData16(Structure):
    _fields_ = [
        ('flag', u16),
        ('padding', u16),
        ('buffer', u32), #POINTER(u16)),
    ]

class PMiSleepCallbackInfo(Structure):
    _fields_ = [
        ('callback', PMSleepCallback),
        ('arg', u32),
        ('next', u32), #POINTER(PMSleepCallbackInfo)),
    ]
