from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.gsXML import *
from private.mkds_python_bindings.generated.sake import *
from private.mkds_python_bindings.generated.sakeRequestInternal import *

SAKEIRequestType_CREATE_RECORD = 0
SAKEIRequestType_UPDATE_RECORD = 1
SAKEIRequestType_DELETE_RECORD = 2
SAKEIRequestType_SEARCH_FOR_RECORDS = 3
SAKEIRequestType_GET_MY_RECORDS = 4
SAKEIRequestType_GET_SPECIFIC_RECORDS = 5
SAKEIRequestType_GET_RANDOM_RECORD = 6
SAKEIRequestType_RATE_RECORD = 7
SAKEIRequestType_GET_RECORD_LIMIT = 8
SAKEIRequestType_GET_RECORD_COUNT = 9

SAKEIRequestType_CREATE_RECORD = 0
SAKEIRequestType_UPDATE_RECORD = 1
SAKEIRequestType_DELETE_RECORD = 2
SAKEIRequestType_SEARCH_FOR_RECORDS = 3
SAKEIRequestType_GET_MY_RECORDS = 4
SAKEIRequestType_GET_SPECIFIC_RECORDS = 5
SAKEIRequestType_GET_RANDOM_RECORD = 6
SAKEIRequestType_RATE_RECORD = 7
SAKEIRequestType_GET_RECORD_LIMIT = 8
SAKEIRequestType_GET_RECORD_COUNT = 9


class SAKERequestInternal(Structure):
    _fields_ = [
        ('mSake', SAKE),
        ('mType', SAKEIRequestType),
        ('mInput', c_void_p32),
        ('mOutput', c_void_p32),
        ('mCallback', SAKERequestCallback),
        ('mUserData', c_void_p32),
        ('mSoapRequest', GSXmlStreamWriter),
        ('mSoapResponse', GSXmlStreamWriter),
        ('mInfo', POINTER32(SAKEIRequestInfo)),
    ]
SAKEIRequestType = c_int
