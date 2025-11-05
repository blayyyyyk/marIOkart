from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.cps import *
from private.mkds_python_bindings.generated.message import *
from private.mkds_python_bindings.generated.ssl import *
from private.mkds_python_bindings.generated.thread import *
from private.mkds_python_bindings.generated.types import *

SOCL_CHECKCOUNT_UDPRCVCB_1 = 0
SOCL_CHECKCOUNT_UDPRCVCB_2 = 1
SOCL_CHECKCOUNT_MAX = 2

SOCL_CHECKCOUNT_UDPRCVCB_1 = 0
SOCL_CHECKCOUNT_UDPRCVCB_2 = 1
SOCL_CHECKCOUNT_MAX = 2

SOCL_CHECKCOUNT_UDPRCVCB_1 = 0
SOCL_CHECKCOUNT_UDPRCVCB_2 = 1
SOCL_CHECKCOUNT_MAX = 2

SOCL_CHECKCOUNT_UDPRCVCB_1 = 0
SOCL_CHECKCOUNT_UDPRCVCB_2 = 1
SOCL_CHECKCOUNT_MAX = 2

SOCLCaInfo = CPSCaInfo

class SOCLSocketCommandPipe(Structure):
    _fields_ = [
        ('queue', OSMessageQueue),
        ('thread', OSThread),
        ('in_use', OSMutex),
    ]

class SOCLiCommandPacket(Union):
    _fields_ = [
        ('h', SOCLiCommandHeader),
        ('create_socket', SOCLiCommandCreateSocket),
        ('bind', SOCLiCommandBind),
        ('listen_accept', SOCLiCommandListenAccept),
        ('read', SOCLiCommandRead),
        ('consume', SOCLiCommandConsume),
        ('write', SOCLiCommandWrite),
        ('shutdown', SOCLiCommandShutdown),
        ('close', SOCLiCommandClose),
        ('enable_ssl', SOCLiCommandEnableSsl),
    ]

class SOCLConfig(Structure):
    _fields_ = [
        ('use_dhcp', BOOL),
        ('host_ip', struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitroWiFi_socl_h_623_5),
        ('alloc', c_void_p32),
        ('free', c_void_p32),
        ('cmd_packet_max', u32),
        ('lan_buffer_size', u32),
        ('lan_buffer', c_void_p32),
        ('cps_thread_prio', s32),
        ('mtu', s32),
        ('rwin', s32),
    ]

class SOCLSocketThreadParam(Structure):
    _fields_ = [
        ('stack_size', u16),
        ('priority', u8),
        ('queue_max', u8),
    ]

class SOCLSocketParam(Structure):
    _fields_ = [
        ('flag_mode', s8),
        ('flag_block', s8),
        ('buffer', SOCLSocketBufferParam),
        ('recv_pipe', SOCLSocketCommandPipeParam),
        ('send_pipe', SOCLSocketCommandPipeParam),
    ]
SOCLSslAuthHandler = c_void_p32

class SOCLiCommandEnableSsl(Structure):
    _fields_ = [
        ('h', SOCLiCommandHeader),
        ('connection', POINTER32(SOCLSslConnection)),
    ]

class SOCLiCommandCreateSocket(Structure):
    _fields_ = [
        ('h', SOCLiCommandHeader),
    ]

class SOCLiCommandWrite(Structure):
    _fields_ = [
        ('h', SOCLiCommandHeader),
        ('buffer1', POINTER32(u8)),
        ('buffer1_len', s32),
        ('buffer2', POINTER32(u8)),
        ('buffer2_len', s32),
        ('wrtbuf_after', u16),
        ('padding', (u8 * 2)),
        ('local_port', u16),
        ('remote_port', u16),
        ('remote_ip', SOCLInAddr),
    ]

class SOCLiCommandShutdown(Structure):
    _fields_ = [
        ('h', SOCLiCommandHeader),
    ]

class SOCLiCommandClose(Structure):
    _fields_ = [
        ('h', SOCLiCommandHeader),
    ]

class SOCLiCommandRead(Structure):
    _fields_ = [
        ('h', SOCLiCommandHeader),
        ('buffer', POINTER32(u8)),
        ('buffer_len', s32),
        ('remote_port', POINTER32(u16)),
        ('remote_ip', POINTER32(SOCLInAddr)),
        ('flag_noconsume', s8),
        ('padding', (u8 * 3)),
    ]

class SOCLiCommandBind(Structure):
    _fields_ = [
        ('h', SOCLiCommandHeader),
        ('local_port', u16),
        ('remote_port', u16),
        ('remote_ip', SOCLInAddr),
    ]

class SOCLiCommandListenAccept(Structure):
    _fields_ = [
        ('h', SOCLiCommandHeader),
        ('local_port', u16),
        ('padding', (u8 * 2)),
        ('remote_port_ptr', POINTER32(u16)),
        ('remote_ip_ptr', POINTER32(SOCLInAddr)),
    ]

class SOCLiCommandConsume(Structure):
    _fields_ = [
        ('h', SOCLiCommandHeader),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitroWiFi_socl_h_623_5(Structure):
    _fields_ = [
        ('my_ip', SOCLInAddr),
        ('net_mask', SOCLInAddr),
        ('gateway_ip', SOCLInAddr),
        ('dns_ip', (SOCLInAddr * 2)),
    ]

class SOCLSocketBufferParam(Structure):
    _fields_ = [
        ('rcvbuf_size', u16),
        ('rcvbuf_consume_min', u16),
        ('sndbuf_size', u16),
        ('linbuf_size', u16),
        ('outbuf_size', u16),
        ('wrtbuf_size', u16),
        ('udpbuf_size', u16),
    ]
SOCLSslConnection = CPSSslConnection

class SOCLiCommandHeader(Structure):
    _fields_ = [
        ('handler', SOCLiCommandHandler),
        ('socket', POINTER32(SOCLSocket)),
        ('response', POINTER32(OSMessageQueue)),
        ('flag_mode', s8),
        ('flag_block', s8),
        ('padding', (u8 * 2)),
    ]
SOCLiCommandHandler = c_void_p32

class SOCLiSocketRingBuffer(Structure):
    _fields_ = [
        ('area', CPSSocBuf),
        ('in', vu16),
        ('out', vu16),
        ('waiting', OSThreadQueue),
    ]

class SOCLiSocketUdpData(Structure):
    _fields_ = [
        ('next', POINTER32(SOCLiSocketUdpData)),
        ('size', u16),
        ('remote_port', u16),
        ('remote_ip', SOCLInAddr),
    ]

class SOCLiSocketUdpDataControl(Structure):
    _fields_ = [
        ('in', POINTER32(SOCLiSocketUdpData)),
        ('out', POINTER32(SOCLiSocketUdpData)),
        ('size', vu16),
        ('size_max', u16),
        ('waiting', OSThreadQueue),
    ]

class SOCLiSocketRecvCommandPipe(Structure):
    _fields_ = [
        ('h', SOCLiSocketCommandPipe),
        ('consumed', vs32),
        ('consumed_min', u16),
        ('flag_noconsume', s8),
        ('padding', (u8 * 1)),
        ('udpdata', SOCLiSocketUdpDataControl),
    ]

class SOCLiSocketSendCommandPipe(Structure):
    _fields_ = [
        ('h', SOCLiSocketCommandPipe),
        ('buffer', SOCLiSocketRingBuffer),
        ('exe_socket', POINTER32(SOCLSocket)),
    ]

class SOCLSocket(Structure):
    _fields_ = [
        ('cps_socket', CPSSoc),
        ('recv_pipe', POINTER32(SOCLiSocketRecvCommandPipe)),
        ('send_pipe', POINTER32(SOCLiSocketSendCommandPipe)),
        ('result', vs32),
        ('state', vs16),
        ('flag_block', s8),
        ('flag_mode', s8),
        ('local_port', u16),
        ('remote_port', u16),
        ('remote_ip', SOCLInAddr),
        ('next', POINTER32(SOCLSocket)),
    ]
SOCLInAddr = c_long
