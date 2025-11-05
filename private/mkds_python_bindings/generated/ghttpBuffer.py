from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.ghttp import *
from private.mkds_python_bindings.generated.ghttpConnection import *


class GHIBuffer(Structure):
    _fields_ = [
        ('connection', POINTER32(GHIConnection)),
        ('data', POINTER32(c_char)),
        ('size', c_int),
        ('len', c_int),
        ('pos', c_int),
        ('sizeIncrement', c_int),
        ('fixed', GHTTPBool),
        ('dontFree', GHTTPBool),
        ('readOnly', GHTTPBool),
    ]
