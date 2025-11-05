from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.darray import *
from private.mkds_python_bindings.generated.ghttp import *

GHIPostingError = 0
GHIPostingDone = 1
GHIPostingPosting = 2
GHIPostingWaitForContinue = 3

GHIPostingError = 0
GHIPostingDone = 1
GHIPostingPosting = 2
GHIPostingWaitForContinue = 3

GHIPostingError = 0
GHIPostingDone = 1
GHIPostingPosting = 2
GHIPostingWaitForContinue = 3

GHIPostingError = 0
GHIPostingDone = 1
GHIPostingPosting = 2
GHIPostingWaitForContinue = 3

GHIPostingError = 0
GHIPostingDone = 1
GHIPostingPosting = 2
GHIPostingWaitForContinue = 3

GHIPostingError = 0
GHIPostingDone = 1
GHIPostingPosting = 2
GHIPostingWaitForContinue = 3

GHIPostingError = 0
GHIPostingDone = 1
GHIPostingPosting = 2
GHIPostingWaitForContinue = 3

GHIPostingError = 0
GHIPostingDone = 1
GHIPostingPosting = 2
GHIPostingWaitForContinue = 3

GHIPostingError = 0
GHIPostingDone = 1
GHIPostingPosting = 2
GHIPostingWaitForContinue = 3

GHIPostingError = 0
GHIPostingDone = 1
GHIPostingPosting = 2
GHIPostingWaitForContinue = 3

GHIPostingResult = c_int

class GHIPostingState(Structure):
    _fields_ = [
        ('states', DArray),
        ('index', c_int),
        ('bytesPosted', c_int),
        ('totalBytes', c_int),
        ('callback', ghttpPostCallback),
        ('param', c_void_p32),
        ('waitPostContinue', GHTTPBool),
        ('completed', GHTTPBool),
    ]
