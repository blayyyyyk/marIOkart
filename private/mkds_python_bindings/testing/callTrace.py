from ctypes import *
from private.mkds_python_bindings.testing.types import *

OS_CALLTRACE_STACK = 0
OS_CALLTRACE_LOG = 1

OS_CALLTRACE_STACK = 0
OS_CALLTRACE_LOG = 1

OSCallTraceMode = c_int


class OSCallTrace(Structure):
    _fields_ = [
        ('name', u32),
        ('returnAddress', u32),
        ('level', u32),
        ('r0', u32),
        ('r1', u32),
        ('r2', u32),
        ('r3', u32),
    ]

class OSCallTraceInfo(Structure):
    _fields_ = [
        ('current', u32), #POINTER(OSCallTrace)),
        ('limit', u32), #POINTER(OSCallTrace)),
        ('enable', u16),
        ('circular', u16),
        ('level', u32),
        ('array', (OSCallTrace * 1)),
    ]
