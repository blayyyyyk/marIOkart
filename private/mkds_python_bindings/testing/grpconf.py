from ctypes import *
from private.mkds_python_bindings.testing.types import *


class grpconf_entry_t(Structure):
    _fields_ = [
        ('objectId', u16),
        ('has3DModel', u16),
        ('nearClip', u16),
        ('farClip', u16),
        ('collisionType', u16),
        ('width', u16),
        ('height', u16),
        ('depth', u16),
    ]
