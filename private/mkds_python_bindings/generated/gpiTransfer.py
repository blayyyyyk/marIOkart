from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.darray import *
from private.mkds_python_bindings.generated.gp import *
from private.mkds_python_bindings.generated.gpi import *
from private.mkds_python_bindings.generated.gpiPeer import *
from private.mkds_python_bindings.generated.gsPlatform import *
from private.mkds_python_bindings.generated.md5 import *

GPITransferPinging = 0
GPITransferWaiting = 1
GPITransferTransferring = 2
GPITransferComplete = 3
GPITransferCancelled = 4
GPITransferNoConnection = 5

GPITransferPinging = 0
GPITransferWaiting = 1
GPITransferTransferring = 2
GPITransferComplete = 3
GPITransferCancelled = 4
GPITransferNoConnection = 5

GPITransferPinging = 0
GPITransferWaiting = 1
GPITransferTransferring = 2
GPITransferComplete = 3
GPITransferCancelled = 4
GPITransferNoConnection = 5

GPITransferPinging = 0
GPITransferWaiting = 1
GPITransferTransferring = 2
GPITransferComplete = 3
GPITransferCancelled = 4
GPITransferNoConnection = 5

GPITransferPinging = 0
GPITransferWaiting = 1
GPITransferTransferring = 2
GPITransferComplete = 3
GPITransferCancelled = 4
GPITransferNoConnection = 5

GPITransferPinging = 0
GPITransferWaiting = 1
GPITransferTransferring = 2
GPITransferComplete = 3
GPITransferCancelled = 4
GPITransferNoConnection = 5

GPITransferPinging = 0
GPITransferWaiting = 1
GPITransferTransferring = 2
GPITransferComplete = 3
GPITransferCancelled = 4
GPITransferNoConnection = 5

GPITransferPinging = 0
GPITransferWaiting = 1
GPITransferTransferring = 2
GPITransferComplete = 3
GPITransferCancelled = 4
GPITransferNoConnection = 5

GPITransferPinging = 0
GPITransferWaiting = 1
GPITransferTransferring = 2
GPITransferComplete = 3
GPITransferCancelled = 4
GPITransferNoConnection = 5

GPITransferPinging = 0
GPITransferWaiting = 1
GPITransferTransferring = 2
GPITransferComplete = 3
GPITransferCancelled = 4
GPITransferNoConnection = 5

GPITransferPinging = 0
GPITransferWaiting = 1
GPITransferTransferring = 2
GPITransferComplete = 3
GPITransferCancelled = 4
GPITransferNoConnection = 5

GPITransferPinging = 0
GPITransferWaiting = 1
GPITransferTransferring = 2
GPITransferComplete = 3
GPITransferCancelled = 4
GPITransferNoConnection = 5

GPITransferPinging = 0
GPITransferWaiting = 1
GPITransferTransferring = 2
GPITransferComplete = 3
GPITransferCancelled = 4
GPITransferNoConnection = 5

GPITransferPinging = 0
GPITransferWaiting = 1
GPITransferTransferring = 2
GPITransferComplete = 3
GPITransferCancelled = 4
GPITransferNoConnection = 5

GPITransferPinging = 0
GPITransferWaiting = 1
GPITransferTransferring = 2
GPITransferComplete = 3
GPITransferCancelled = 4
GPITransferNoConnection = 5

GPITransferPinging = 0
GPITransferWaiting = 1
GPITransferTransferring = 2
GPITransferComplete = 3
GPITransferCancelled = 4
GPITransferNoConnection = 5

GPITransferPinging = 0
GPITransferWaiting = 1
GPITransferTransferring = 2
GPITransferComplete = 3
GPITransferCancelled = 4
GPITransferNoConnection = 5

GPITransferPinging = 0
GPITransferWaiting = 1
GPITransferTransferring = 2
GPITransferComplete = 3
GPITransferCancelled = 4
GPITransferNoConnection = 5

GPITransferPinging = 0
GPITransferWaiting = 1
GPITransferTransferring = 2
GPITransferComplete = 3
GPITransferCancelled = 4
GPITransferNoConnection = 5

GPITransferPinging = 0
GPITransferWaiting = 1
GPITransferTransferring = 2
GPITransferComplete = 3
GPITransferCancelled = 4
GPITransferNoConnection = 5

GPITransferPinging = 0
GPITransferWaiting = 1
GPITransferTransferring = 2
GPITransferComplete = 3
GPITransferCancelled = 4
GPITransferNoConnection = 5

GPITransferPinging = 0
GPITransferWaiting = 1
GPITransferTransferring = 2
GPITransferComplete = 3
GPITransferCancelled = 4
GPITransferNoConnection = 5

GPITransferPinging = 0
GPITransferWaiting = 1
GPITransferTransferring = 2
GPITransferComplete = 3
GPITransferCancelled = 4
GPITransferNoConnection = 5

GPITransferPinging = 0
GPITransferWaiting = 1
GPITransferTransferring = 2
GPITransferComplete = 3
GPITransferCancelled = 4
GPITransferNoConnection = 5

GPITransferPinging = 0
GPITransferWaiting = 1
GPITransferTransferring = 2
GPITransferComplete = 3
GPITransferCancelled = 4
GPITransferNoConnection = 5

GPITransferPinging = 0
GPITransferWaiting = 1
GPITransferTransferring = 2
GPITransferComplete = 3
GPITransferCancelled = 4
GPITransferNoConnection = 5

GPITransferPinging = 0
GPITransferWaiting = 1
GPITransferTransferring = 2
GPITransferComplete = 3
GPITransferCancelled = 4
GPITransferNoConnection = 5

GPITransferPinging = 0
GPITransferWaiting = 1
GPITransferTransferring = 2
GPITransferComplete = 3
GPITransferCancelled = 4
GPITransferNoConnection = 5


class GPITransferID_s(Structure):
    _fields_ = [
        ('profileid', c_int),
        ('count', c_int),
        ('time', c_int),
    ]

class GPITransfer(Structure):
    _fields_ = [
        ('state', GPITransferState),
        ('files', DArray),
        ('transferID', GPITransferID),
        ('localID', c_int),
        ('sender', GPIBool),
        ('profile', GPProfile),
        ('peer', POINTER32(GPIPeer)),
        ('currentFile', c_int),
        ('throttle', c_int),
        ('baseDirectory', POINTER32(c_char)),
        ('lastSend', c_long),
        ('message', POINTER32(c_char)),
        ('totalSize', c_int),
        ('progress', c_int),
        ('userData', c_void_p32),
    ]

class GPIFile(Structure):
    _fields_ = [
        ('path', POINTER32(c_char)),
        ('name', POINTER32(c_char)),
        ('progress', c_int),
        ('size', c_int),
        ('acknowledged', c_int),
        ('file', POINTER32(FILE)),
        ('flags', c_int),
        ('modTime', gsi_time),
        ('md5', MD5_CTX),
        ('reason', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_dwc_gs_gp_gpiTransfer_h_60_9(Structure):
    _fields_ = [
        ('state', GPITransferState),
        ('files', DArray),
        ('transferID', GPITransferID),
        ('localID', c_int),
        ('sender', GPIBool),
        ('profile', GPProfile),
        ('peer', POINTER32(c_int)),
        ('currentFile', c_int),
        ('throttle', c_int),
        ('baseDirectory', POINTER32(c_char)),
        ('lastSend', c_long),
        ('message', POINTER32(c_char)),
        ('totalSize', c_int),
        ('progress', c_int),
        ('userData', c_void_p32),
    ]
GPITransferState = c_int
