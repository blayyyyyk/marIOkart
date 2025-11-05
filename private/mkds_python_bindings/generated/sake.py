from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.gsPlatform import *

SAKEStartupResult_SUCCESS = 0
SAKEStartupResult_NOT_AVAILABLE = 1
SAKEStartupResult_CORE_SHUTDOWN = 2
SAKEStartupResult_OUT_OF_MEMORY = 3

SAKEStartupResult_SUCCESS = 0
SAKEStartupResult_NOT_AVAILABLE = 1
SAKEStartupResult_CORE_SHUTDOWN = 2
SAKEStartupResult_OUT_OF_MEMORY = 3

SAKEFieldType_BYTE = 0
SAKEFieldType_SHORT = 1
SAKEFieldType_INT = 2
SAKEFieldType_FLOAT = 3
SAKEFieldType_ASCII_STRING = 4
SAKEFieldType_UNICODE_STRING = 5
SAKEFieldType_BOOLEAN = 6
SAKEFieldType_DATE_AND_TIME = 7
SAKEFieldType_BINARY_DATA = 8
SAKEFieldType_NUM_FIELD_TYPES = 9

SAKEFieldType_BYTE = 0
SAKEFieldType_SHORT = 1
SAKEFieldType_INT = 2
SAKEFieldType_FLOAT = 3
SAKEFieldType_ASCII_STRING = 4
SAKEFieldType_UNICODE_STRING = 5
SAKEFieldType_BOOLEAN = 6
SAKEFieldType_DATE_AND_TIME = 7
SAKEFieldType_BINARY_DATA = 8
SAKEFieldType_NUM_FIELD_TYPES = 9

SAKEStartRequestResult_SUCCESS = 0
SAKEStartRequestResult_NOT_AUTHENTICATED = 1
SAKEStartRequestResult_OUT_OF_MEMORY = 2
SAKEStartRequestResult_BAD_INPUT = 3
SAKEStartRequestResult_BAD_TABLEID = 4
SAKEStartRequestResult_BAD_FIELDS = 5
SAKEStartRequestResult_BAD_NUM_FIELDS = 6
SAKEStartRequestResult_BAD_FIELD_NAME = 7
SAKEStartRequestResult_BAD_FIELD_TYPE = 8
SAKEStartRequestResult_BAD_FIELD_VALUE = 9
SAKEStartRequestResult_BAD_OFFSET = 10
SAKEStartRequestResult_BAD_MAX = 11
SAKEStartRequestResult_BAD_RECORDIDS = 12
SAKEStartRequestResult_BAD_NUM_RECORDIDS = 13
SAKEStartRequestResult_UNKNOWN_ERROR = 14

SAKEStartRequestResult_SUCCESS = 0
SAKEStartRequestResult_NOT_AUTHENTICATED = 1
SAKEStartRequestResult_OUT_OF_MEMORY = 2
SAKEStartRequestResult_BAD_INPUT = 3
SAKEStartRequestResult_BAD_TABLEID = 4
SAKEStartRequestResult_BAD_FIELDS = 5
SAKEStartRequestResult_BAD_NUM_FIELDS = 6
SAKEStartRequestResult_BAD_FIELD_NAME = 7
SAKEStartRequestResult_BAD_FIELD_TYPE = 8
SAKEStartRequestResult_BAD_FIELD_VALUE = 9
SAKEStartRequestResult_BAD_OFFSET = 10
SAKEStartRequestResult_BAD_MAX = 11
SAKEStartRequestResult_BAD_RECORDIDS = 12
SAKEStartRequestResult_BAD_NUM_RECORDIDS = 13
SAKEStartRequestResult_UNKNOWN_ERROR = 14

SAKERequestResult_SUCCESS = 0
SAKERequestResult_SECRET_KEY_INVALID = 1
SAKERequestResult_SERVICE_DISABLED = 2
SAKERequestResult_CONNECTION_TIMEOUT = 3
SAKERequestResult_CONNECTION_ERROR = 4
SAKERequestResult_MALFORMED_RESPONSE = 5
SAKERequestResult_OUT_OF_MEMORY = 6
SAKERequestResult_DATABASE_UNAVAILABLE = 7
SAKERequestResult_LOGIN_TICKET_INVALID = 8
SAKERequestResult_LOGIN_TICKET_EXPIRED = 9
SAKERequestResult_TABLE_NOT_FOUND = 10
SAKERequestResult_RECORD_NOT_FOUND = 11
SAKERequestResult_FIELD_NOT_FOUND = 12
SAKERequestResult_FIELD_TYPE_INVALID = 13
SAKERequestResult_NO_PERMISSION = 14
SAKERequestResult_RECORD_LIMIT_REACHED = 15
SAKERequestResult_ALREADY_RATED = 16
SAKERequestResult_NOT_RATEABLE = 17
SAKERequestResult_NOT_OWNED = 18
SAKERequestResult_FILTER_INVALID = 19
SAKERequestResult_SORT_INVALID = 20
SAKERequestResult_TARGET_FILTER_INVALID = 21
SAKERequestResult_UNKNOWN_ERROR = 22

SAKERequestResult_SUCCESS = 0
SAKERequestResult_SECRET_KEY_INVALID = 1
SAKERequestResult_SERVICE_DISABLED = 2
SAKERequestResult_CONNECTION_TIMEOUT = 3
SAKERequestResult_CONNECTION_ERROR = 4
SAKERequestResult_MALFORMED_RESPONSE = 5
SAKERequestResult_OUT_OF_MEMORY = 6
SAKERequestResult_DATABASE_UNAVAILABLE = 7
SAKERequestResult_LOGIN_TICKET_INVALID = 8
SAKERequestResult_LOGIN_TICKET_EXPIRED = 9
SAKERequestResult_TABLE_NOT_FOUND = 10
SAKERequestResult_RECORD_NOT_FOUND = 11
SAKERequestResult_FIELD_NOT_FOUND = 12
SAKERequestResult_FIELD_TYPE_INVALID = 13
SAKERequestResult_NO_PERMISSION = 14
SAKERequestResult_RECORD_LIMIT_REACHED = 15
SAKERequestResult_ALREADY_RATED = 16
SAKERequestResult_NOT_RATEABLE = 17
SAKERequestResult_NOT_OWNED = 18
SAKERequestResult_FILTER_INVALID = 19
SAKERequestResult_SORT_INVALID = 20
SAKERequestResult_TARGET_FILTER_INVALID = 21
SAKERequestResult_UNKNOWN_ERROR = 22

SAKEFileResult_SUCCESS = 0
SAKEFileResult_BAD_HTTP_METHOD = 1
SAKEFileResult_BAD_FILE_COUNT = 2
SAKEFileResult_MISSING_PARAMETER = 3
SAKEFileResult_FILE_NOT_FOUND = 4
SAKEFileResult_FILE_TOO_LARGE = 5
SAKEFileResult_SERVER_ERROR = 6
SAKEFileResult_UNKNOWN_ERROR = 7

SAKEFileResult_SUCCESS = 0
SAKEFileResult_BAD_HTTP_METHOD = 1
SAKEFileResult_BAD_FILE_COUNT = 2
SAKEFileResult_MISSING_PARAMETER = 3
SAKEFileResult_FILE_NOT_FOUND = 4
SAKEFileResult_FILE_TOO_LARGE = 5
SAKEFileResult_SERVER_ERROR = 6
SAKEFileResult_UNKNOWN_ERROR = 7

SAKEStartupResult_SUCCESS = 0
SAKEStartupResult_NOT_AVAILABLE = 1
SAKEStartupResult_CORE_SHUTDOWN = 2
SAKEStartupResult_OUT_OF_MEMORY = 3

SAKEStartupResult_SUCCESS = 0
SAKEStartupResult_NOT_AVAILABLE = 1
SAKEStartupResult_CORE_SHUTDOWN = 2
SAKEStartupResult_OUT_OF_MEMORY = 3

SAKEFieldType_BYTE = 0
SAKEFieldType_SHORT = 1
SAKEFieldType_INT = 2
SAKEFieldType_FLOAT = 3
SAKEFieldType_ASCII_STRING = 4
SAKEFieldType_UNICODE_STRING = 5
SAKEFieldType_BOOLEAN = 6
SAKEFieldType_DATE_AND_TIME = 7
SAKEFieldType_BINARY_DATA = 8
SAKEFieldType_NUM_FIELD_TYPES = 9

SAKEFieldType_BYTE = 0
SAKEFieldType_SHORT = 1
SAKEFieldType_INT = 2
SAKEFieldType_FLOAT = 3
SAKEFieldType_ASCII_STRING = 4
SAKEFieldType_UNICODE_STRING = 5
SAKEFieldType_BOOLEAN = 6
SAKEFieldType_DATE_AND_TIME = 7
SAKEFieldType_BINARY_DATA = 8
SAKEFieldType_NUM_FIELD_TYPES = 9

SAKEStartRequestResult_SUCCESS = 0
SAKEStartRequestResult_NOT_AUTHENTICATED = 1
SAKEStartRequestResult_OUT_OF_MEMORY = 2
SAKEStartRequestResult_BAD_INPUT = 3
SAKEStartRequestResult_BAD_TABLEID = 4
SAKEStartRequestResult_BAD_FIELDS = 5
SAKEStartRequestResult_BAD_NUM_FIELDS = 6
SAKEStartRequestResult_BAD_FIELD_NAME = 7
SAKEStartRequestResult_BAD_FIELD_TYPE = 8
SAKEStartRequestResult_BAD_FIELD_VALUE = 9
SAKEStartRequestResult_BAD_OFFSET = 10
SAKEStartRequestResult_BAD_MAX = 11
SAKEStartRequestResult_BAD_RECORDIDS = 12
SAKEStartRequestResult_BAD_NUM_RECORDIDS = 13
SAKEStartRequestResult_UNKNOWN_ERROR = 14

SAKEStartRequestResult_SUCCESS = 0
SAKEStartRequestResult_NOT_AUTHENTICATED = 1
SAKEStartRequestResult_OUT_OF_MEMORY = 2
SAKEStartRequestResult_BAD_INPUT = 3
SAKEStartRequestResult_BAD_TABLEID = 4
SAKEStartRequestResult_BAD_FIELDS = 5
SAKEStartRequestResult_BAD_NUM_FIELDS = 6
SAKEStartRequestResult_BAD_FIELD_NAME = 7
SAKEStartRequestResult_BAD_FIELD_TYPE = 8
SAKEStartRequestResult_BAD_FIELD_VALUE = 9
SAKEStartRequestResult_BAD_OFFSET = 10
SAKEStartRequestResult_BAD_MAX = 11
SAKEStartRequestResult_BAD_RECORDIDS = 12
SAKEStartRequestResult_BAD_NUM_RECORDIDS = 13
SAKEStartRequestResult_UNKNOWN_ERROR = 14

SAKERequestResult_SUCCESS = 0
SAKERequestResult_SECRET_KEY_INVALID = 1
SAKERequestResult_SERVICE_DISABLED = 2
SAKERequestResult_CONNECTION_TIMEOUT = 3
SAKERequestResult_CONNECTION_ERROR = 4
SAKERequestResult_MALFORMED_RESPONSE = 5
SAKERequestResult_OUT_OF_MEMORY = 6
SAKERequestResult_DATABASE_UNAVAILABLE = 7
SAKERequestResult_LOGIN_TICKET_INVALID = 8
SAKERequestResult_LOGIN_TICKET_EXPIRED = 9
SAKERequestResult_TABLE_NOT_FOUND = 10
SAKERequestResult_RECORD_NOT_FOUND = 11
SAKERequestResult_FIELD_NOT_FOUND = 12
SAKERequestResult_FIELD_TYPE_INVALID = 13
SAKERequestResult_NO_PERMISSION = 14
SAKERequestResult_RECORD_LIMIT_REACHED = 15
SAKERequestResult_ALREADY_RATED = 16
SAKERequestResult_NOT_RATEABLE = 17
SAKERequestResult_NOT_OWNED = 18
SAKERequestResult_FILTER_INVALID = 19
SAKERequestResult_SORT_INVALID = 20
SAKERequestResult_TARGET_FILTER_INVALID = 21
SAKERequestResult_UNKNOWN_ERROR = 22

SAKERequestResult_SUCCESS = 0
SAKERequestResult_SECRET_KEY_INVALID = 1
SAKERequestResult_SERVICE_DISABLED = 2
SAKERequestResult_CONNECTION_TIMEOUT = 3
SAKERequestResult_CONNECTION_ERROR = 4
SAKERequestResult_MALFORMED_RESPONSE = 5
SAKERequestResult_OUT_OF_MEMORY = 6
SAKERequestResult_DATABASE_UNAVAILABLE = 7
SAKERequestResult_LOGIN_TICKET_INVALID = 8
SAKERequestResult_LOGIN_TICKET_EXPIRED = 9
SAKERequestResult_TABLE_NOT_FOUND = 10
SAKERequestResult_RECORD_NOT_FOUND = 11
SAKERequestResult_FIELD_NOT_FOUND = 12
SAKERequestResult_FIELD_TYPE_INVALID = 13
SAKERequestResult_NO_PERMISSION = 14
SAKERequestResult_RECORD_LIMIT_REACHED = 15
SAKERequestResult_ALREADY_RATED = 16
SAKERequestResult_NOT_RATEABLE = 17
SAKERequestResult_NOT_OWNED = 18
SAKERequestResult_FILTER_INVALID = 19
SAKERequestResult_SORT_INVALID = 20
SAKERequestResult_TARGET_FILTER_INVALID = 21
SAKERequestResult_UNKNOWN_ERROR = 22

SAKEFileResult_SUCCESS = 0
SAKEFileResult_BAD_HTTP_METHOD = 1
SAKEFileResult_BAD_FILE_COUNT = 2
SAKEFileResult_MISSING_PARAMETER = 3
SAKEFileResult_FILE_NOT_FOUND = 4
SAKEFileResult_FILE_TOO_LARGE = 5
SAKEFileResult_SERVER_ERROR = 6
SAKEFileResult_UNKNOWN_ERROR = 7

SAKEFileResult_SUCCESS = 0
SAKEFileResult_BAD_HTTP_METHOD = 1
SAKEFileResult_BAD_FILE_COUNT = 2
SAKEFileResult_MISSING_PARAMETER = 3
SAKEFileResult_FILE_NOT_FOUND = 4
SAKEFileResult_FILE_TOO_LARGE = 5
SAKEFileResult_SERVER_ERROR = 6
SAKEFileResult_UNKNOWN_ERROR = 7

SAKEStartupResult_SUCCESS = 0
SAKEStartupResult_NOT_AVAILABLE = 1
SAKEStartupResult_CORE_SHUTDOWN = 2
SAKEStartupResult_OUT_OF_MEMORY = 3

SAKEStartupResult_SUCCESS = 0
SAKEStartupResult_NOT_AVAILABLE = 1
SAKEStartupResult_CORE_SHUTDOWN = 2
SAKEStartupResult_OUT_OF_MEMORY = 3

SAKEFieldType_BYTE = 0
SAKEFieldType_SHORT = 1
SAKEFieldType_INT = 2
SAKEFieldType_FLOAT = 3
SAKEFieldType_ASCII_STRING = 4
SAKEFieldType_UNICODE_STRING = 5
SAKEFieldType_BOOLEAN = 6
SAKEFieldType_DATE_AND_TIME = 7
SAKEFieldType_BINARY_DATA = 8
SAKEFieldType_NUM_FIELD_TYPES = 9

SAKEFieldType_BYTE = 0
SAKEFieldType_SHORT = 1
SAKEFieldType_INT = 2
SAKEFieldType_FLOAT = 3
SAKEFieldType_ASCII_STRING = 4
SAKEFieldType_UNICODE_STRING = 5
SAKEFieldType_BOOLEAN = 6
SAKEFieldType_DATE_AND_TIME = 7
SAKEFieldType_BINARY_DATA = 8
SAKEFieldType_NUM_FIELD_TYPES = 9

SAKEStartRequestResult_SUCCESS = 0
SAKEStartRequestResult_NOT_AUTHENTICATED = 1
SAKEStartRequestResult_OUT_OF_MEMORY = 2
SAKEStartRequestResult_BAD_INPUT = 3
SAKEStartRequestResult_BAD_TABLEID = 4
SAKEStartRequestResult_BAD_FIELDS = 5
SAKEStartRequestResult_BAD_NUM_FIELDS = 6
SAKEStartRequestResult_BAD_FIELD_NAME = 7
SAKEStartRequestResult_BAD_FIELD_TYPE = 8
SAKEStartRequestResult_BAD_FIELD_VALUE = 9
SAKEStartRequestResult_BAD_OFFSET = 10
SAKEStartRequestResult_BAD_MAX = 11
SAKEStartRequestResult_BAD_RECORDIDS = 12
SAKEStartRequestResult_BAD_NUM_RECORDIDS = 13
SAKEStartRequestResult_UNKNOWN_ERROR = 14

SAKEStartRequestResult_SUCCESS = 0
SAKEStartRequestResult_NOT_AUTHENTICATED = 1
SAKEStartRequestResult_OUT_OF_MEMORY = 2
SAKEStartRequestResult_BAD_INPUT = 3
SAKEStartRequestResult_BAD_TABLEID = 4
SAKEStartRequestResult_BAD_FIELDS = 5
SAKEStartRequestResult_BAD_NUM_FIELDS = 6
SAKEStartRequestResult_BAD_FIELD_NAME = 7
SAKEStartRequestResult_BAD_FIELD_TYPE = 8
SAKEStartRequestResult_BAD_FIELD_VALUE = 9
SAKEStartRequestResult_BAD_OFFSET = 10
SAKEStartRequestResult_BAD_MAX = 11
SAKEStartRequestResult_BAD_RECORDIDS = 12
SAKEStartRequestResult_BAD_NUM_RECORDIDS = 13
SAKEStartRequestResult_UNKNOWN_ERROR = 14

SAKERequestResult_SUCCESS = 0
SAKERequestResult_SECRET_KEY_INVALID = 1
SAKERequestResult_SERVICE_DISABLED = 2
SAKERequestResult_CONNECTION_TIMEOUT = 3
SAKERequestResult_CONNECTION_ERROR = 4
SAKERequestResult_MALFORMED_RESPONSE = 5
SAKERequestResult_OUT_OF_MEMORY = 6
SAKERequestResult_DATABASE_UNAVAILABLE = 7
SAKERequestResult_LOGIN_TICKET_INVALID = 8
SAKERequestResult_LOGIN_TICKET_EXPIRED = 9
SAKERequestResult_TABLE_NOT_FOUND = 10
SAKERequestResult_RECORD_NOT_FOUND = 11
SAKERequestResult_FIELD_NOT_FOUND = 12
SAKERequestResult_FIELD_TYPE_INVALID = 13
SAKERequestResult_NO_PERMISSION = 14
SAKERequestResult_RECORD_LIMIT_REACHED = 15
SAKERequestResult_ALREADY_RATED = 16
SAKERequestResult_NOT_RATEABLE = 17
SAKERequestResult_NOT_OWNED = 18
SAKERequestResult_FILTER_INVALID = 19
SAKERequestResult_SORT_INVALID = 20
SAKERequestResult_TARGET_FILTER_INVALID = 21
SAKERequestResult_UNKNOWN_ERROR = 22

SAKERequestResult_SUCCESS = 0
SAKERequestResult_SECRET_KEY_INVALID = 1
SAKERequestResult_SERVICE_DISABLED = 2
SAKERequestResult_CONNECTION_TIMEOUT = 3
SAKERequestResult_CONNECTION_ERROR = 4
SAKERequestResult_MALFORMED_RESPONSE = 5
SAKERequestResult_OUT_OF_MEMORY = 6
SAKERequestResult_DATABASE_UNAVAILABLE = 7
SAKERequestResult_LOGIN_TICKET_INVALID = 8
SAKERequestResult_LOGIN_TICKET_EXPIRED = 9
SAKERequestResult_TABLE_NOT_FOUND = 10
SAKERequestResult_RECORD_NOT_FOUND = 11
SAKERequestResult_FIELD_NOT_FOUND = 12
SAKERequestResult_FIELD_TYPE_INVALID = 13
SAKERequestResult_NO_PERMISSION = 14
SAKERequestResult_RECORD_LIMIT_REACHED = 15
SAKERequestResult_ALREADY_RATED = 16
SAKERequestResult_NOT_RATEABLE = 17
SAKERequestResult_NOT_OWNED = 18
SAKERequestResult_FILTER_INVALID = 19
SAKERequestResult_SORT_INVALID = 20
SAKERequestResult_TARGET_FILTER_INVALID = 21
SAKERequestResult_UNKNOWN_ERROR = 22

SAKEFileResult_SUCCESS = 0
SAKEFileResult_BAD_HTTP_METHOD = 1
SAKEFileResult_BAD_FILE_COUNT = 2
SAKEFileResult_MISSING_PARAMETER = 3
SAKEFileResult_FILE_NOT_FOUND = 4
SAKEFileResult_FILE_TOO_LARGE = 5
SAKEFileResult_SERVER_ERROR = 6
SAKEFileResult_UNKNOWN_ERROR = 7

SAKEFileResult_SUCCESS = 0
SAKEFileResult_BAD_HTTP_METHOD = 1
SAKEFileResult_BAD_FILE_COUNT = 2
SAKEFileResult_MISSING_PARAMETER = 3
SAKEFileResult_FILE_NOT_FOUND = 4
SAKEFileResult_FILE_TOO_LARGE = 5
SAKEFileResult_SERVER_ERROR = 6
SAKEFileResult_UNKNOWN_ERROR = 7

SAKEStartupResult_SUCCESS = 0
SAKEStartupResult_NOT_AVAILABLE = 1
SAKEStartupResult_CORE_SHUTDOWN = 2
SAKEStartupResult_OUT_OF_MEMORY = 3

SAKEStartupResult_SUCCESS = 0
SAKEStartupResult_NOT_AVAILABLE = 1
SAKEStartupResult_CORE_SHUTDOWN = 2
SAKEStartupResult_OUT_OF_MEMORY = 3

SAKEFieldType_BYTE = 0
SAKEFieldType_SHORT = 1
SAKEFieldType_INT = 2
SAKEFieldType_FLOAT = 3
SAKEFieldType_ASCII_STRING = 4
SAKEFieldType_UNICODE_STRING = 5
SAKEFieldType_BOOLEAN = 6
SAKEFieldType_DATE_AND_TIME = 7
SAKEFieldType_BINARY_DATA = 8
SAKEFieldType_NUM_FIELD_TYPES = 9

SAKEFieldType_BYTE = 0
SAKEFieldType_SHORT = 1
SAKEFieldType_INT = 2
SAKEFieldType_FLOAT = 3
SAKEFieldType_ASCII_STRING = 4
SAKEFieldType_UNICODE_STRING = 5
SAKEFieldType_BOOLEAN = 6
SAKEFieldType_DATE_AND_TIME = 7
SAKEFieldType_BINARY_DATA = 8
SAKEFieldType_NUM_FIELD_TYPES = 9

SAKEStartRequestResult_SUCCESS = 0
SAKEStartRequestResult_NOT_AUTHENTICATED = 1
SAKEStartRequestResult_OUT_OF_MEMORY = 2
SAKEStartRequestResult_BAD_INPUT = 3
SAKEStartRequestResult_BAD_TABLEID = 4
SAKEStartRequestResult_BAD_FIELDS = 5
SAKEStartRequestResult_BAD_NUM_FIELDS = 6
SAKEStartRequestResult_BAD_FIELD_NAME = 7
SAKEStartRequestResult_BAD_FIELD_TYPE = 8
SAKEStartRequestResult_BAD_FIELD_VALUE = 9
SAKEStartRequestResult_BAD_OFFSET = 10
SAKEStartRequestResult_BAD_MAX = 11
SAKEStartRequestResult_BAD_RECORDIDS = 12
SAKEStartRequestResult_BAD_NUM_RECORDIDS = 13
SAKEStartRequestResult_UNKNOWN_ERROR = 14

SAKEStartRequestResult_SUCCESS = 0
SAKEStartRequestResult_NOT_AUTHENTICATED = 1
SAKEStartRequestResult_OUT_OF_MEMORY = 2
SAKEStartRequestResult_BAD_INPUT = 3
SAKEStartRequestResult_BAD_TABLEID = 4
SAKEStartRequestResult_BAD_FIELDS = 5
SAKEStartRequestResult_BAD_NUM_FIELDS = 6
SAKEStartRequestResult_BAD_FIELD_NAME = 7
SAKEStartRequestResult_BAD_FIELD_TYPE = 8
SAKEStartRequestResult_BAD_FIELD_VALUE = 9
SAKEStartRequestResult_BAD_OFFSET = 10
SAKEStartRequestResult_BAD_MAX = 11
SAKEStartRequestResult_BAD_RECORDIDS = 12
SAKEStartRequestResult_BAD_NUM_RECORDIDS = 13
SAKEStartRequestResult_UNKNOWN_ERROR = 14

SAKERequestResult_SUCCESS = 0
SAKERequestResult_SECRET_KEY_INVALID = 1
SAKERequestResult_SERVICE_DISABLED = 2
SAKERequestResult_CONNECTION_TIMEOUT = 3
SAKERequestResult_CONNECTION_ERROR = 4
SAKERequestResult_MALFORMED_RESPONSE = 5
SAKERequestResult_OUT_OF_MEMORY = 6
SAKERequestResult_DATABASE_UNAVAILABLE = 7
SAKERequestResult_LOGIN_TICKET_INVALID = 8
SAKERequestResult_LOGIN_TICKET_EXPIRED = 9
SAKERequestResult_TABLE_NOT_FOUND = 10
SAKERequestResult_RECORD_NOT_FOUND = 11
SAKERequestResult_FIELD_NOT_FOUND = 12
SAKERequestResult_FIELD_TYPE_INVALID = 13
SAKERequestResult_NO_PERMISSION = 14
SAKERequestResult_RECORD_LIMIT_REACHED = 15
SAKERequestResult_ALREADY_RATED = 16
SAKERequestResult_NOT_RATEABLE = 17
SAKERequestResult_NOT_OWNED = 18
SAKERequestResult_FILTER_INVALID = 19
SAKERequestResult_SORT_INVALID = 20
SAKERequestResult_TARGET_FILTER_INVALID = 21
SAKERequestResult_UNKNOWN_ERROR = 22

SAKERequestResult_SUCCESS = 0
SAKERequestResult_SECRET_KEY_INVALID = 1
SAKERequestResult_SERVICE_DISABLED = 2
SAKERequestResult_CONNECTION_TIMEOUT = 3
SAKERequestResult_CONNECTION_ERROR = 4
SAKERequestResult_MALFORMED_RESPONSE = 5
SAKERequestResult_OUT_OF_MEMORY = 6
SAKERequestResult_DATABASE_UNAVAILABLE = 7
SAKERequestResult_LOGIN_TICKET_INVALID = 8
SAKERequestResult_LOGIN_TICKET_EXPIRED = 9
SAKERequestResult_TABLE_NOT_FOUND = 10
SAKERequestResult_RECORD_NOT_FOUND = 11
SAKERequestResult_FIELD_NOT_FOUND = 12
SAKERequestResult_FIELD_TYPE_INVALID = 13
SAKERequestResult_NO_PERMISSION = 14
SAKERequestResult_RECORD_LIMIT_REACHED = 15
SAKERequestResult_ALREADY_RATED = 16
SAKERequestResult_NOT_RATEABLE = 17
SAKERequestResult_NOT_OWNED = 18
SAKERequestResult_FILTER_INVALID = 19
SAKERequestResult_SORT_INVALID = 20
SAKERequestResult_TARGET_FILTER_INVALID = 21
SAKERequestResult_UNKNOWN_ERROR = 22

SAKEFileResult_SUCCESS = 0
SAKEFileResult_BAD_HTTP_METHOD = 1
SAKEFileResult_BAD_FILE_COUNT = 2
SAKEFileResult_MISSING_PARAMETER = 3
SAKEFileResult_FILE_NOT_FOUND = 4
SAKEFileResult_FILE_TOO_LARGE = 5
SAKEFileResult_SERVER_ERROR = 6
SAKEFileResult_UNKNOWN_ERROR = 7

SAKEFileResult_SUCCESS = 0
SAKEFileResult_BAD_HTTP_METHOD = 1
SAKEFileResult_BAD_FILE_COUNT = 2
SAKEFileResult_MISSING_PARAMETER = 3
SAKEFileResult_FILE_NOT_FOUND = 4
SAKEFileResult_FILE_TOO_LARGE = 5
SAKEFileResult_SERVER_ERROR = 6
SAKEFileResult_UNKNOWN_ERROR = 7


class SAKECreateRecordInput(Structure):
    _fields_ = [
        ('mTableId', POINTER32(c_char)),
        ('mFields', POINTER32(SAKEField)),
        ('mNumFields', c_int),
    ]

class SAKECreateRecordOutput(Structure):
    _fields_ = [
        ('mRecordId', c_int),
    ]

class SAKEUpdateRecordInput(Structure):
    _fields_ = [
        ('mTableId', POINTER32(c_char)),
        ('mRecordId', c_int),
        ('mFields', POINTER32(SAKEField)),
        ('mNumFields', c_int),
    ]

class SAKEDeleteRecordInput(Structure):
    _fields_ = [
        ('mTableId', POINTER32(c_char)),
        ('mRecordId', c_int),
    ]

class SAKESearchForRecordsInput(Structure):
    _fields_ = [
        ('mTableId', POINTER32(c_char)),
        ('mFieldNames', POINTER32(POINTER32(c_char))),
        ('mNumFields', c_int),
        ('mFilter', POINTER32(c_char)),
        ('mSort', POINTER32(c_char)),
        ('mOffset', c_int),
        ('mMaxRecords', c_int),
        ('mTargetRecordFilter', POINTER32(c_char)),
        ('mSurroundingRecordsCount', c_int),
        ('mOwnerIds', POINTER32(c_int)),
        ('mNumOwnerIds', c_int),
        ('mCacheFlag', gsi_bool),
    ]

class SAKESearchForRecordsOutput(Structure):
    _fields_ = [
        ('mNumRecords', c_int),
        ('mRecords', POINTER32(POINTER32(SAKEField))),
    ]

class SAKEGetMyRecordsInput(Structure):
    _fields_ = [
        ('mTableId', POINTER32(c_char)),
        ('mFieldNames', POINTER32(POINTER32(c_char))),
        ('mNumFields', c_int),
    ]

class SAKEGetMyRecordsOutput(Structure):
    _fields_ = [
        ('mNumRecords', c_int),
        ('mRecords', POINTER32(POINTER32(SAKEField))),
    ]

class SAKEGetSpecificRecordsInput(Structure):
    _fields_ = [
        ('mTableId', POINTER32(c_char)),
        ('mRecordIds', POINTER32(c_int)),
        ('mNumRecordIds', c_int),
        ('mFieldNames', POINTER32(POINTER32(c_char))),
        ('mNumFields', c_int),
    ]

class SAKEGetSpecificRecordsOutput(Structure):
    _fields_ = [
        ('mNumRecords', c_int),
        ('mRecords', POINTER32(POINTER32(SAKEField))),
    ]

class SAKEGetRandomRecordInput(Structure):
    _fields_ = [
        ('mTableId', POINTER32(c_char)),
        ('mFieldNames', POINTER32(POINTER32(c_char))),
        ('mNumFields', c_int),
        ('mFilter', POINTER32(c_char)),
    ]

class SAKEGetRandomRecordOutput(Structure):
    _fields_ = [
        ('mRecord', POINTER32(SAKEField)),
    ]

class SAKERateRecordInput(Structure):
    _fields_ = [
        ('mTableId', POINTER32(c_char)),
        ('mRecordId', c_int),
        ('mRating', gsi_u8),
    ]

class SAKERateRecordOutput(Structure):
    _fields_ = [
        ('mNumRatings', c_int),
        ('mAverageRating', c_float),
    ]

class SAKEGetRecordLimitInput(Structure):
    _fields_ = [
        ('mTableId', POINTER32(c_char)),
    ]

class SAKEGetRecordLimitOutput(Structure):
    _fields_ = [
        ('mLimitPerOwner', c_int),
        ('mNumOwned', c_int),
    ]

class SAKEGetRecordCountInput(Structure):
    _fields_ = [
        ('mTableId', POINTER32(c_char)),
        ('mFilter', POINTER32(c_char)),
        ('mCacheFlag', gsi_bool),
    ]

class SAKEGetRecordCountOutput(Structure):
    _fields_ = [
        ('mCount', c_int),
    ]
SAKEStartupResult = c_int
SAKEFileResult = c_int
SAKERequestResult = c_int
SAKERequest = c_void_p32

class SAKEField(Structure):
    _fields_ = [
        ('mName', POINTER32(c_char)),
        ('mType', SAKEFieldType),
        ('mValue', SAKEValue),
    ]
SAKEStartRequestResult = c_int
SAKE = c_void_p32
SAKERequestCallback = c_void_p32
SAKEFieldType = c_int

class SAKEValue(Union):
    _fields_ = [
        ('mByte', gsi_u8),
        ('mShort', gsi_i16),
        ('mInt', gsi_i32),
        ('mFloat', c_float),
        ('mAsciiString', POINTER32(c_char)),
        ('mUnicodeString', POINTER32(gsi_u16)),
        ('mBoolean', gsi_bool),
        ('mDateAndTime', time_t),
        ('mBinaryData', SAKEBinaryData),
    ]

class SAKEBinaryData(Structure):
    _fields_ = [
        ('mValue', POINTER32(gsi_u8)),
        ('mLength', c_int),
    ]
