from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
GHIRecvData = 0
GHINoData = 1
GHIConnClosed = 2
GHIError = 3

GHIRecvData = 0
GHINoData = 1
GHIConnClosed = 2
GHIError = 3

GHITrySendError = 0
GHITrySendSent = 1
GHITrySendBuffered = 2

GHITrySendError = 0
GHITrySendSent = 1
GHITrySendBuffered = 2

GHIRecvResult = c_int
GHITrySendResult = c_int
