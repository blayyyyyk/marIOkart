from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.thread import *
from private.mkds_python_bindings.generated.tick import *
from private.mkds_python_bindings.generated.types import *


class OSiAlarm(Structure):
    _fields_ = [
        ('handler', OSAlarmHandler),
        ('arg', c_void_p32),
        ('tag', u32),
        ('fire', OSTick),
        ('prev', POINTER32(OSAlarm)),
        ('next', POINTER32(OSAlarm)),
        ('period', OSTick),
        ('start', OSTick),
    ]

class OSiAlarmQueue(Structure):
    _fields_ = [
        ('head', POINTER32(OSAlarm)),
        ('tail', POINTER32(OSAlarm)),
    ]
SNDAlarmHandler = c_void_p32
OSAlarmHandler = c_void_p32

class SNDAlarm(Structure):
    _fields_ = [
        ('enable', u8),
        ('id', u8),
        ('padding', u16),
        ('setting', struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_snd_common_alarm_h_83_5),
        ('alarm', OSAlarm),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_snd_common_alarm_h_83_5(Structure):
    _fields_ = [
        ('tick', OSTick),
        ('period', OSTick),
    ]
