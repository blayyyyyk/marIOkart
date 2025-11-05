from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.ghttp import *
from private.mkds_python_bindings.generated.gsCore import *
from private.mkds_python_bindings.generated.gsPlatform import *
from private.mkds_python_bindings.generated.gsXML import *


class GSSoapTask(Structure):
    _fields_ = [
        ('mCallbackFunc', GSSoapCallbackFunc),
        ('mCustomFunc', GSSoapCustomFunc),
        ('mURL', POINTER32(c_char)),
        ('mService', POINTER32(c_char)),
        ('mRequestSoap', GSXmlStreamWriter),
        ('mResponseSoap', GSXmlStreamReader),
        ('mResponseBuffer', POINTER32(c_char)),
        ('mPostData', GHTTPPost),
        ('mUserData', c_void_p32),
        ('mCoreTask', POINTER32(GSTask)),
        ('mRequestId', GHTTPRequest),
        ('mRequestResult', GHTTPResult),
        ('mCompleted', gsi_bool),
    ]
GSSoapCustomFunc = c_void_p32
GSSoapCallbackFunc = c_void_p32
