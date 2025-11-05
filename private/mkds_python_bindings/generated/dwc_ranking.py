from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
DWC_RNK_SUCCESS = 0
DWC_RNK_IN_ERROR = 1
DWC_RNK_ERROR_INVALID_PARAMETER = 2
DWC_RNK_ERROR_INIT_ALREADYINITIALIZED = 3
DWC_RNK_ERROR_INIT_INVALID_INITDATASIZE = 4
DWC_RNK_ERROR_INIT_INVALID_INITDATA = 5
DWC_RNK_ERROR_INIT_INVALID_USERDATA = 6
DWC_RNK_ERROR_PUT_NOTREADY = 7
DWC_RNK_ERROR_PUT_INVALID_KEY = 8
DWC_RNK_ERROR_PUT_NOMEMORY = 9
DWC_RNK_ERROR_GET_NOTREADY = 10
DWC_RNK_ERROR_GET_INVALID_KEY = 11
DWC_RNK_ERROR_GET_NOMEMORY = 12
DWC_RNK_ERROR_CANCEL_NOTASK = 13
DWC_RNK_PROCESS_NOTASK = 14
DWC_RNK_PROCESS_TIMEOUT = 15
DWC_RNK_ERROR_INVALID_MODE = 16
DWC_RNK_ERROR_NOTCOMPLETED = 17
DWC_RNK_ERROR_EMPTY_RESPONSE = 18

DWC_RNK_SUCCESS = 0
DWC_RNK_IN_ERROR = 1
DWC_RNK_ERROR_INVALID_PARAMETER = 2
DWC_RNK_ERROR_INIT_ALREADYINITIALIZED = 3
DWC_RNK_ERROR_INIT_INVALID_INITDATASIZE = 4
DWC_RNK_ERROR_INIT_INVALID_INITDATA = 5
DWC_RNK_ERROR_INIT_INVALID_USERDATA = 6
DWC_RNK_ERROR_PUT_NOTREADY = 7
DWC_RNK_ERROR_PUT_INVALID_KEY = 8
DWC_RNK_ERROR_PUT_NOMEMORY = 9
DWC_RNK_ERROR_GET_NOTREADY = 10
DWC_RNK_ERROR_GET_INVALID_KEY = 11
DWC_RNK_ERROR_GET_NOMEMORY = 12
DWC_RNK_ERROR_CANCEL_NOTASK = 13
DWC_RNK_PROCESS_NOTASK = 14
DWC_RNK_PROCESS_TIMEOUT = 15
DWC_RNK_ERROR_INVALID_MODE = 16
DWC_RNK_ERROR_NOTCOMPLETED = 17
DWC_RNK_ERROR_EMPTY_RESPONSE = 18

DWC_RNK_STATE_NOTREADY = 0
DWC_RNK_STATE_INITIALIZED = 1
DWC_RNK_STATE_PUT_ASYNC = 2
DWC_RNK_STATE_GET_ASYNC = 3
DWC_RNK_STATE_COMPLETED = 4
DWC_RNK_STATE_TIMEOUT = 5
DWC_RNK_STATE_ERROR = 6

DWC_RNK_STATE_NOTREADY = 0
DWC_RNK_STATE_INITIALIZED = 1
DWC_RNK_STATE_PUT_ASYNC = 2
DWC_RNK_STATE_GET_ASYNC = 3
DWC_RNK_STATE_COMPLETED = 4
DWC_RNK_STATE_TIMEOUT = 5
DWC_RNK_STATE_ERROR = 6

DWC_RNK_REGION_JP = 1
DWC_RNK_REGION_US = 2
DWC_RNK_REGION_EU = 4
DWC_RNK_REGION_KR = 8
DWC_RNK_REGION_ALL = 255

DWC_RNK_REGION_JP = 1
DWC_RNK_REGION_US = 2
DWC_RNK_REGION_EU = 4
DWC_RNK_REGION_KR = 8
DWC_RNK_REGION_ALL = 255

DWC_RNK_GET_MODE_ORDER = 0
DWC_RNK_GET_MODE_TOPLIST = 1
DWC_RNK_GET_MODE_NEAR = 2
DWC_RNK_GET_MODE_FRIENDS = 3
DWC_RNK_GET_MODE_NEAR_HI = 4
DWC_RNK_GET_MODE_NEAR_LOW = 5

DWC_RNK_GET_MODE_ORDER = 0
DWC_RNK_GET_MODE_TOPLIST = 1
DWC_RNK_GET_MODE_NEAR = 2
DWC_RNK_GET_MODE_FRIENDS = 3
DWC_RNK_GET_MODE_NEAR_HI = 4
DWC_RNK_GET_MODE_NEAR_LOW = 5

DWC_RNK_SUCCESS = 0
DWC_RNK_IN_ERROR = 1
DWC_RNK_ERROR_INVALID_PARAMETER = 2
DWC_RNK_ERROR_INIT_ALREADYINITIALIZED = 3
DWC_RNK_ERROR_INIT_INVALID_INITDATASIZE = 4
DWC_RNK_ERROR_INIT_INVALID_INITDATA = 5
DWC_RNK_ERROR_INIT_INVALID_USERDATA = 6
DWC_RNK_ERROR_PUT_NOTREADY = 7
DWC_RNK_ERROR_PUT_INVALID_KEY = 8
DWC_RNK_ERROR_PUT_NOMEMORY = 9
DWC_RNK_ERROR_GET_NOTREADY = 10
DWC_RNK_ERROR_GET_INVALID_KEY = 11
DWC_RNK_ERROR_GET_NOMEMORY = 12
DWC_RNK_ERROR_CANCEL_NOTASK = 13
DWC_RNK_PROCESS_NOTASK = 14
DWC_RNK_PROCESS_TIMEOUT = 15
DWC_RNK_ERROR_INVALID_MODE = 16
DWC_RNK_ERROR_NOTCOMPLETED = 17
DWC_RNK_ERROR_EMPTY_RESPONSE = 18

DWC_RNK_SUCCESS = 0
DWC_RNK_IN_ERROR = 1
DWC_RNK_ERROR_INVALID_PARAMETER = 2
DWC_RNK_ERROR_INIT_ALREADYINITIALIZED = 3
DWC_RNK_ERROR_INIT_INVALID_INITDATASIZE = 4
DWC_RNK_ERROR_INIT_INVALID_INITDATA = 5
DWC_RNK_ERROR_INIT_INVALID_USERDATA = 6
DWC_RNK_ERROR_PUT_NOTREADY = 7
DWC_RNK_ERROR_PUT_INVALID_KEY = 8
DWC_RNK_ERROR_PUT_NOMEMORY = 9
DWC_RNK_ERROR_GET_NOTREADY = 10
DWC_RNK_ERROR_GET_INVALID_KEY = 11
DWC_RNK_ERROR_GET_NOMEMORY = 12
DWC_RNK_ERROR_CANCEL_NOTASK = 13
DWC_RNK_PROCESS_NOTASK = 14
DWC_RNK_PROCESS_TIMEOUT = 15
DWC_RNK_ERROR_INVALID_MODE = 16
DWC_RNK_ERROR_NOTCOMPLETED = 17
DWC_RNK_ERROR_EMPTY_RESPONSE = 18

DWC_RNK_STATE_NOTREADY = 0
DWC_RNK_STATE_INITIALIZED = 1
DWC_RNK_STATE_PUT_ASYNC = 2
DWC_RNK_STATE_GET_ASYNC = 3
DWC_RNK_STATE_COMPLETED = 4
DWC_RNK_STATE_TIMEOUT = 5
DWC_RNK_STATE_ERROR = 6

DWC_RNK_STATE_NOTREADY = 0
DWC_RNK_STATE_INITIALIZED = 1
DWC_RNK_STATE_PUT_ASYNC = 2
DWC_RNK_STATE_GET_ASYNC = 3
DWC_RNK_STATE_COMPLETED = 4
DWC_RNK_STATE_TIMEOUT = 5
DWC_RNK_STATE_ERROR = 6

DWC_RNK_REGION_JP = 1
DWC_RNK_REGION_US = 2
DWC_RNK_REGION_EU = 4
DWC_RNK_REGION_KR = 8
DWC_RNK_REGION_ALL = 255

DWC_RNK_REGION_JP = 1
DWC_RNK_REGION_US = 2
DWC_RNK_REGION_EU = 4
DWC_RNK_REGION_KR = 8
DWC_RNK_REGION_ALL = 255

DWC_RNK_GET_MODE_ORDER = 0
DWC_RNK_GET_MODE_TOPLIST = 1
DWC_RNK_GET_MODE_NEAR = 2
DWC_RNK_GET_MODE_FRIENDS = 3
DWC_RNK_GET_MODE_NEAR_HI = 4
DWC_RNK_GET_MODE_NEAR_LOW = 5

DWC_RNK_GET_MODE_ORDER = 0
DWC_RNK_GET_MODE_TOPLIST = 1
DWC_RNK_GET_MODE_NEAR = 2
DWC_RNK_GET_MODE_FRIENDS = 3
DWC_RNK_GET_MODE_NEAR_HI = 4
DWC_RNK_GET_MODE_NEAR_LOW = 5


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_dwc_base_dwc_ranking_h_212_9(Structure):
    _fields_ = [
        ('order', c_int),
        ('pid', c_int),
        ('score', c_int),
        ('region', DWCRnkRegion),
        ('lastupdate', c_int),
        ('size', c_int),
        ('userdata', c_void_p32),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_dwc_base_dwc_ranking_h_228_9(Structure):
    _fields_ = [
        ('size', c_int),
    ]

class union__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_dwc_base_dwc_ranking_h_232_2(Union):
    _fields_ = [
        ('order', struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_dwc_base_dwc_ranking_h_237_3),
        ('toplist', struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_dwc_base_dwc_ranking_h_248_3),
        ('near', struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_dwc_base_dwc_ranking_h_260_3),
        ('friends', struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_dwc_base_dwc_ranking_h_272_3),
        ('data', c_int),
    ]
DWCRnkError = c_int
DWCRnkState = c_int
DWCRnkGetMode = c_int
DWCRnkRegion = c_int

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_dwc_base_dwc_ranking_h_260_3(Structure):
    _fields_ = [
        ('sort', c_int),
        ('limit', c_int),
        ('since', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_dwc_base_dwc_ranking_h_248_3(Structure):
    _fields_ = [
        ('sort', c_int),
        ('limit', c_int),
        ('since', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_dwc_base_dwc_ranking_h_272_3(Structure):
    _fields_ = [
        ('sort', c_int),
        ('limit', c_int),
        ('since', c_int),
        ('friends', (c_int * 64)),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_dwc_base_dwc_ranking_h_237_3(Structure):
    _fields_ = [
        ('sort', c_int),
        ('since', c_int),
    ]
