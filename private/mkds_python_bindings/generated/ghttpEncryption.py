from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.ghttp import *

GHIEncryptionMethod_None = 0
GHIEncryptionMethod_Encrypt = 1
GHIEncryptionMethod_Decrypt = 2

GHIEncryptionMethod_None = 0
GHIEncryptionMethod_Encrypt = 1
GHIEncryptionMethod_Decrypt = 2

GHIEncryptionResult_None = 0
GHIEncryptionResult_Success = 1
GHIEncryptionResult_BufferTooSmall = 2
GHIEncryptionResult_Error = 3

GHIEncryptionResult_None = 0
GHIEncryptionResult_Success = 1
GHIEncryptionResult_BufferTooSmall = 2
GHIEncryptionResult_Error = 3

GHIEncryptionMethod_None = 0
GHIEncryptionMethod_Encrypt = 1
GHIEncryptionMethod_Decrypt = 2

GHIEncryptionMethod_None = 0
GHIEncryptionMethod_Encrypt = 1
GHIEncryptionMethod_Decrypt = 2

GHIEncryptionResult_None = 0
GHIEncryptionResult_Success = 1
GHIEncryptionResult_BufferTooSmall = 2
GHIEncryptionResult_Error = 3

GHIEncryptionResult_None = 0
GHIEncryptionResult_Success = 1
GHIEncryptionResult_BufferTooSmall = 2
GHIEncryptionResult_Error = 3

GHIEncryptionMethod_None = 0
GHIEncryptionMethod_Encrypt = 1
GHIEncryptionMethod_Decrypt = 2

GHIEncryptionMethod_None = 0
GHIEncryptionMethod_Encrypt = 1
GHIEncryptionMethod_Decrypt = 2

GHIEncryptionResult_None = 0
GHIEncryptionResult_Success = 1
GHIEncryptionResult_BufferTooSmall = 2
GHIEncryptionResult_Error = 3

GHIEncryptionResult_None = 0
GHIEncryptionResult_Success = 1
GHIEncryptionResult_BufferTooSmall = 2
GHIEncryptionResult_Error = 3

GHIEncryptionMethod_None = 0
GHIEncryptionMethod_Encrypt = 1
GHIEncryptionMethod_Decrypt = 2

GHIEncryptionMethod_None = 0
GHIEncryptionMethod_Encrypt = 1
GHIEncryptionMethod_Decrypt = 2

GHIEncryptionResult_None = 0
GHIEncryptionResult_Success = 1
GHIEncryptionResult_BufferTooSmall = 2
GHIEncryptionResult_Error = 3

GHIEncryptionResult_None = 0
GHIEncryptionResult_Success = 1
GHIEncryptionResult_BufferTooSmall = 2
GHIEncryptionResult_Error = 3

GHIEncryptionMethod_None = 0
GHIEncryptionMethod_Encrypt = 1
GHIEncryptionMethod_Decrypt = 2

GHIEncryptionMethod_None = 0
GHIEncryptionMethod_Encrypt = 1
GHIEncryptionMethod_Decrypt = 2

GHIEncryptionResult_None = 0
GHIEncryptionResult_Success = 1
GHIEncryptionResult_BufferTooSmall = 2
GHIEncryptionResult_Error = 3

GHIEncryptionResult_None = 0
GHIEncryptionResult_Success = 1
GHIEncryptionResult_BufferTooSmall = 2
GHIEncryptionResult_Error = 3

GHIEncryptionMethod_None = 0
GHIEncryptionMethod_Encrypt = 1
GHIEncryptionMethod_Decrypt = 2

GHIEncryptionMethod_None = 0
GHIEncryptionMethod_Encrypt = 1
GHIEncryptionMethod_Decrypt = 2

GHIEncryptionResult_None = 0
GHIEncryptionResult_Success = 1
GHIEncryptionResult_BufferTooSmall = 2
GHIEncryptionResult_Error = 3

GHIEncryptionResult_None = 0
GHIEncryptionResult_Success = 1
GHIEncryptionResult_BufferTooSmall = 2
GHIEncryptionResult_Error = 3

GHIEncryptionMethod_None = 0
GHIEncryptionMethod_Encrypt = 1
GHIEncryptionMethod_Decrypt = 2

GHIEncryptionMethod_None = 0
GHIEncryptionMethod_Encrypt = 1
GHIEncryptionMethod_Decrypt = 2

GHIEncryptionResult_None = 0
GHIEncryptionResult_Success = 1
GHIEncryptionResult_BufferTooSmall = 2
GHIEncryptionResult_Error = 3

GHIEncryptionResult_None = 0
GHIEncryptionResult_Success = 1
GHIEncryptionResult_BufferTooSmall = 2
GHIEncryptionResult_Error = 3

GHIEncryptionMethod = c_int
GHIEncryptionResult = c_int

class GHIEncryptor(Structure):
    _fields_ = [
        ('mInterface', c_void_p32),
        ('mEngine', GHTTPEncryptionEngine),
        ('mInitialized', GHTTPBool),
        ('mSessionEstablished', GHTTPBool),
        ('mInitFunc', GHTTPEncryptorInitFunc),
        ('mCleanupFunc', GHTTPEncryptorCleanupFunc),
        ('mEncryptFunc', GHTTPEncryptorEncryptFunc),
        ('mDecryptFunc', GHTTPEncryptorDecryptFunc),
    ]
GHTTPEncryptorCleanupFunc = c_void_p32
GHTTPEncryptorDecryptFunc = c_void_p32
GHTTPEncryptorInitFunc = c_void_p32
GHTTPEncryptorEncryptFunc = c_void_p32
