from ctypes import *
from private.mkds_python_bindings.testing.tick import *
from private.mkds_python_bindings.testing.types import *

PAD_LOGIC_OR = 0
PAD_LOGIC_AND = 32768
PAD_OR_INTR = 0
PAD_AND_INTR = 32768

PAD_LOGIC_OR = 0
PAD_LOGIC_AND = 32768
PAD_OR_INTR = 0
PAD_AND_INTR = 32768

PADLogic = c_int


class input_pad_t(Structure):
    _fields_ = [
        ('triggeredKeys', u16),
        ('pressedKeys', u16),
        ('releasedKeys', u16),
        ('repeatedKeys', u16),
        ('repeatState', u16),
        ('repeatFrameCounter', u16),
        ('repeatMask', u16),
        ('repeatFirstFrame', u16),
        ('repeatNextFrame', u16),
        ('resetInvoked', u16),
        ('field14', u16),
        ('resetStartTime', OSTick),
        ('field20', u32),
        ('field24', u32),
    ]
