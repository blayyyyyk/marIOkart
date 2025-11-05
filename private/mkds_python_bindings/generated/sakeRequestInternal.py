from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32

class SAKEIRequestInfo(Structure):
    _fields_ = [
        ('mSakeOutputSize', size_t),
        ('mFuncName', POINTER32(c_char)),
        ('mSoapAction', POINTER32(c_char)),
        ('mResponseTag', POINTER32(c_char)),
        ('mResultTag', POINTER32(c_char)),
        ('mValidateInputFunc', c_void_p32),
        ('mFillSoapRequestFunc', c_void_p32),
        ('mProcessSoapResponseFunc', c_void_p32),
        ('mFreeDataFunc', c_void_p32),
    ]
