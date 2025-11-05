from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.thread import *
from private.mkds_python_bindings.generated.types import *
from private.mkds_python_bindings.generated.util_alloc import *

DWCHTTP_E_NOERR = 0
DWCHTTP_E_MEMERR = 1
DWCHTTP_E_DNSERR = 2
DWCHTTP_E_CONERR = 3
DWCHTTP_E_SENDTOUT = 4
DWCHTTP_E_SENDERR = 5
DWCHTTP_E_RECVTOUT = 6
DWCHTTP_E_ABORT = 7
DWCHTTP_E_FINISH = 8
DWCHTTP_E_MAX = 9

DWCHTTP_E_NOERR = 0
DWCHTTP_E_MEMERR = 1
DWCHTTP_E_DNSERR = 2
DWCHTTP_E_CONERR = 3
DWCHTTP_E_SENDTOUT = 4
DWCHTTP_E_SENDERR = 5
DWCHTTP_E_RECVTOUT = 6
DWCHTTP_E_ABORT = 7
DWCHTTP_E_FINISH = 8
DWCHTTP_E_MAX = 9

DWCHTTP_POST = 0
DWCHTTP_GET = 1

DWCHTTP_POST = 0
DWCHTTP_GET = 1

DWCHTTP_E_NOERR = 0
DWCHTTP_E_MEMERR = 1
DWCHTTP_E_DNSERR = 2
DWCHTTP_E_CONERR = 3
DWCHTTP_E_SENDTOUT = 4
DWCHTTP_E_SENDERR = 5
DWCHTTP_E_RECVTOUT = 6
DWCHTTP_E_ABORT = 7
DWCHTTP_E_FINISH = 8
DWCHTTP_E_MAX = 9

DWCHTTP_E_NOERR = 0
DWCHTTP_E_MEMERR = 1
DWCHTTP_E_DNSERR = 2
DWCHTTP_E_CONERR = 3
DWCHTTP_E_SENDTOUT = 4
DWCHTTP_E_SENDERR = 5
DWCHTTP_E_RECVTOUT = 6
DWCHTTP_E_ABORT = 7
DWCHTTP_E_FINISH = 8
DWCHTTP_E_MAX = 9

DWCHTTP_POST = 0
DWCHTTP_GET = 1

DWCHTTP_POST = 0
DWCHTTP_GET = 1

DWCHTTP_E_NOERR = 0
DWCHTTP_E_MEMERR = 1
DWCHTTP_E_DNSERR = 2
DWCHTTP_E_CONERR = 3
DWCHTTP_E_SENDTOUT = 4
DWCHTTP_E_SENDERR = 5
DWCHTTP_E_RECVTOUT = 6
DWCHTTP_E_ABORT = 7
DWCHTTP_E_FINISH = 8
DWCHTTP_E_MAX = 9

DWCHTTP_E_NOERR = 0
DWCHTTP_E_MEMERR = 1
DWCHTTP_E_DNSERR = 2
DWCHTTP_E_CONERR = 3
DWCHTTP_E_SENDTOUT = 4
DWCHTTP_E_SENDERR = 5
DWCHTTP_E_RECVTOUT = 6
DWCHTTP_E_ABORT = 7
DWCHTTP_E_FINISH = 8
DWCHTTP_E_MAX = 9

DWCHTTP_POST = 0
DWCHTTP_GET = 1

DWCHTTP_POST = 0
DWCHTTP_GET = 1

DWCHTTP_E_NOERR = 0
DWCHTTP_E_MEMERR = 1
DWCHTTP_E_DNSERR = 2
DWCHTTP_E_CONERR = 3
DWCHTTP_E_SENDTOUT = 4
DWCHTTP_E_SENDERR = 5
DWCHTTP_E_RECVTOUT = 6
DWCHTTP_E_ABORT = 7
DWCHTTP_E_FINISH = 8
DWCHTTP_E_MAX = 9

DWCHTTP_E_NOERR = 0
DWCHTTP_E_MEMERR = 1
DWCHTTP_E_DNSERR = 2
DWCHTTP_E_CONERR = 3
DWCHTTP_E_SENDTOUT = 4
DWCHTTP_E_SENDERR = 5
DWCHTTP_E_RECVTOUT = 6
DWCHTTP_E_ABORT = 7
DWCHTTP_E_FINISH = 8
DWCHTTP_E_MAX = 9

DWCHTTP_POST = 0
DWCHTTP_GET = 1

DWCHTTP_POST = 0
DWCHTTP_GET = 1


class DWCHttpParseResult(Structure):
    _fields_ = [
        ('entry', POINTER32(DWCHttpLabelValue)),
        ('len', c_int),
        ('index', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_dwc_auth_dwc_http_h_182_9(Structure):
    _fields_ = [
        ('stack', (c_char * 4096)),
        ('initflag', u8),
        ('param', DWCHttpParam),
        ('error', DWCHttpError),
        ('url', (c_char * 256)),
        ('hostname', POINTER32(c_char)),
        ('filepath', POINTER32(c_char)),
        ('hostip', u32),
        ('ssl_enabled', BOOL),
        ('port', c_short),
        ('soc', c_int),
        ('con', c_int),
        ('lowrecvbuf', POINTER32(c_char)),
        ('lowsendbuf', POINTER32(c_char)),
        ('lowentropydata', (u32 * 4)),
        ('num_postitem', c_int),
        ('req', DWCHttpBuffer),
        ('rep', DWCHttpBuffer),
        ('content_len_mutex', OSMutex),
        ('content_len', c_int),
        ('receivedbody_len', c_int),
        ('labelvalue', (DWCHttpLabelValue * 32)),
        ('thread', OSThread),
        ('mutex', OSMutex),
        ('abort', BOOL),
    ]
DWCHttpError = c_int

class DWCHttpBuffer(Structure):
    _fields_ = [
        ('buffer', POINTER32(c_char)),
        ('write_index', POINTER32(c_char)),
        ('buffer_tail', POINTER32(c_char)),
        ('length', c_long),
    ]

class DWCHttpLabelValue(Structure):
    _fields_ = [
        ('label', POINTER32(c_char)),
        ('value', POINTER32(c_char)),
    ]

class DWCHttpParam(Structure):
    _fields_ = [
        ('url', POINTER32(c_char)),
        ('action', DWCHttpAction),
        ('len_recvbuf', c_long),
        ('alloc', DWCAuthAlloc),
        ('free', DWCAuthFree),
        ('ignoreca', BOOL),
        ('timeout', c_int),
    ]
DWCHttpAction = c_int
