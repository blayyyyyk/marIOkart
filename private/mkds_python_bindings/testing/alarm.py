from ctypes import *
from private.mkds_python_bindings.testing.thread import *
from private.mkds_python_bindings.testing.tick import *
from private.mkds_python_bindings.testing.types import *

OSAlarmHandler = u32
SNDAlarmHandler = u32

osi_alarm_p = u32

class OSiAlarm(Structure):
    _fields_ = [
        ('handler', OSAlarmHandler),
        ('arg', u32),
        ('tag', u32),
        ('fire', OSTick),
        ('prev', osi_alarm_p),
        ('next', osi_alarm_p),
        ('period', OSTick),
        ('start', OSTick),
    ]

class OSiAlarmQueue(Structure):
    _fields_ = [
        ('head', osi_alarm_p),
        ('tail', osi_alarm_p),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_snd_common_alarm_h_83_5_(Structure):
    _fields_ = [
        ('tick', OSTick),
        ('period', OSTick),
    ]

class SNDAlarm(Structure):
    _fields_ = [
        ('enable', u8),
        ('id', u8),
        ('padding', u16),
        ('setting', struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_snd_common_alarm_h_83_5_),
        ('alarm', OSiAlarm),
    ]


