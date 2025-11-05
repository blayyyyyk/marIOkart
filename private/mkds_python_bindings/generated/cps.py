from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.types import *

CPS_STT_CLOSED = 0
CPS_STT_LISTEN = 1
CPS_STT_SYN_SENT = 2
CPS_STT_SYN_RCVD = 3
CPS_STT_ESTABLISHED = 4
CPS_STT_CLOSE_WAIT = 5
CPS_STT_LAST_ACK = 6
CPS_STT_FIN_WAIT1 = 7
CPS_STT_FIN_WAIT2 = 8
CPS_STT_CLOSING = 9
CPS_STT_DATAGRAM = 10
CPS_STT_PING = 11

CPS_BLOCK_NONE = 0
CPS_BLOCK_TCPCON = 1
CPS_BLOCK_TCPREAD = 2
CPS_BLOCK_UDPREAD = 3

CPS_NOIP_REASON_NONE = 0
CPS_NOIP_REASON_LINKOFF = 1
CPS_NOIP_REASON_DHCPDISCOVERY = 2
CPS_NOIP_REASON_LEASETIMEOUT = 3
CPS_NOIP_REASON_COLLISION = 4

CPS_STT_CLOSED = 0
CPS_STT_LISTEN = 1
CPS_STT_SYN_SENT = 2
CPS_STT_SYN_RCVD = 3
CPS_STT_ESTABLISHED = 4
CPS_STT_CLOSE_WAIT = 5
CPS_STT_LAST_ACK = 6
CPS_STT_FIN_WAIT1 = 7
CPS_STT_FIN_WAIT2 = 8
CPS_STT_CLOSING = 9
CPS_STT_DATAGRAM = 10
CPS_STT_PING = 11

CPS_BLOCK_NONE = 0
CPS_BLOCK_TCPCON = 1
CPS_BLOCK_TCPREAD = 2
CPS_BLOCK_UDPREAD = 3

CPS_NOIP_REASON_NONE = 0
CPS_NOIP_REASON_LINKOFF = 1
CPS_NOIP_REASON_DHCPDISCOVERY = 2
CPS_NOIP_REASON_LEASETIMEOUT = 3
CPS_NOIP_REASON_COLLISION = 4

CPS_STT_CLOSED = 0
CPS_STT_LISTEN = 1
CPS_STT_SYN_SENT = 2
CPS_STT_SYN_RCVD = 3
CPS_STT_ESTABLISHED = 4
CPS_STT_CLOSE_WAIT = 5
CPS_STT_LAST_ACK = 6
CPS_STT_FIN_WAIT1 = 7
CPS_STT_FIN_WAIT2 = 8
CPS_STT_CLOSING = 9
CPS_STT_DATAGRAM = 10
CPS_STT_PING = 11

CPS_BLOCK_NONE = 0
CPS_BLOCK_TCPCON = 1
CPS_BLOCK_TCPREAD = 2
CPS_BLOCK_UDPREAD = 3

CPS_NOIP_REASON_NONE = 0
CPS_NOIP_REASON_LINKOFF = 1
CPS_NOIP_REASON_DHCPDISCOVERY = 2
CPS_NOIP_REASON_LEASETIMEOUT = 3
CPS_NOIP_REASON_COLLISION = 4

CPS_STT_CLOSED = 0
CPS_STT_LISTEN = 1
CPS_STT_SYN_SENT = 2
CPS_STT_SYN_RCVD = 3
CPS_STT_ESTABLISHED = 4
CPS_STT_CLOSE_WAIT = 5
CPS_STT_LAST_ACK = 6
CPS_STT_FIN_WAIT1 = 7
CPS_STT_FIN_WAIT2 = 8
CPS_STT_CLOSING = 9
CPS_STT_DATAGRAM = 10
CPS_STT_PING = 11

CPS_BLOCK_NONE = 0
CPS_BLOCK_TCPCON = 1
CPS_BLOCK_TCPREAD = 2
CPS_BLOCK_UDPREAD = 3

CPS_NOIP_REASON_NONE = 0
CPS_NOIP_REASON_LINKOFF = 1
CPS_NOIP_REASON_DHCPDISCOVERY = 2
CPS_NOIP_REASON_LEASETIMEOUT = 3
CPS_NOIP_REASON_COLLISION = 4

CPS_STT_CLOSED = 0
CPS_STT_LISTEN = 1
CPS_STT_SYN_SENT = 2
CPS_STT_SYN_RCVD = 3
CPS_STT_ESTABLISHED = 4
CPS_STT_CLOSE_WAIT = 5
CPS_STT_LAST_ACK = 6
CPS_STT_FIN_WAIT1 = 7
CPS_STT_FIN_WAIT2 = 8
CPS_STT_CLOSING = 9
CPS_STT_DATAGRAM = 10
CPS_STT_PING = 11

CPS_BLOCK_NONE = 0
CPS_BLOCK_TCPCON = 1
CPS_BLOCK_TCPREAD = 2
CPS_BLOCK_UDPREAD = 3

CPS_NOIP_REASON_NONE = 0
CPS_NOIP_REASON_LINKOFF = 1
CPS_NOIP_REASON_DHCPDISCOVERY = 2
CPS_NOIP_REASON_LEASETIMEOUT = 3
CPS_NOIP_REASON_COLLISION = 4


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitroWiFi_cps_h_34_9(Structure):
    _fields_ = [
        ('size', c_int),
        ('data', POINTER32(c_int)),
    ]

class _CPSSoc(Structure):
    _fields_ = [
        ('thread', POINTER32(c_int)),
        ('block_type', c_int),
        ('state', c_int),
        ('ssl', c_int),
        ('local_port', c_int),
        ('con', c_void_p32),
        ('when', c_int),
        ('local_ip_real', c_int),
        ('remote_port', c_int),
        ('remote_port_bound', c_int),
        ('remote_ip', CPSInAddr),
        ('remote_ip_bound', CPSInAddr),
        ('ackno', c_int),
        ('seqno', c_int),
        ('remote_win', c_int),
        ('remote_mss', c_int),
        ('remote_ackno', c_int),
        ('ackrcvd', c_int),
        ('udpread_callback', c_void_p32),
        ('rcvbuf', CPSSocBuf),
        ('rcvbufp', c_int),
        ('sndbuf', CPSSocBuf),
        ('linbuf', CPSSocBuf),
        ('outbuf', CPSSocBuf),
        ('outbufp', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitroWiFi_cps_h_113_9(Structure):
    _fields_ = [
        ('ip', CPSInAddr),
        ('mac', CPSMacAddress),
        ('when', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitroWiFi_cps_h_126_9(Structure):
    _fields_ = [
        ('ipfrom', CPSInAddr),
        ('frags', c_int),
        ('id', c_int),
        ('last', c_int),
        ('size', c_int),
        ('from', (c_int * 8)),
        ('to', (c_int * 8)),
        ('when', c_int),
        ('ofs0', POINTER32(c_int)),
        ('buf', POINTER32(c_int)),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitroWiFi_cps_h_142_9(Structure):
    _fields_ = [
        ('mode', c_int),
        ('alloc', c_void_p32),
        ('free', c_void_p32),
        ('dhcp_callback', c_void_p32),
        ('BOOL', c_void_p32),
        ('random_seed', c_long),
        ('lan_buf', POINTER32(c_int)),
        ('lan_buflen', c_int),
        ('mymss', c_int),
        ('requested_ip', CPSInAddr),
        ('yield_wait', c_int),
    ]

class CPSArpCache(Structure):
    _fields_ = [
        ('ip', CPSInAddr),
        ('mac', CPSMacAddress),
        ('when', u16),
    ]

class CPSFragTable(Structure):
    _fields_ = [
        ('ipfrom', CPSInAddr),
        ('frags', u16),
        ('id', u16),
        ('last', u16),
        ('size', u16),
        ('from', (u16 * 8)),
        ('to', (u16 * 8)),
        ('when', u32),
        ('ofs0', POINTER32(u8)),
        ('buf', POINTER32(u8)),
    ]

class CPSConfig(Structure):
    _fields_ = [
        ('mode', u32),
        ('alloc', c_void_p32),
        ('free', c_void_p32),
        ('dhcp_callback', c_void_p32),
        ('link_is_on', c_void_p32),
        ('random_seed', c_long),
        ('lan_buf', POINTER32(u8)),
        ('lan_buflen', u32),
        ('mymss', u32),
        ('requested_ip', CPSInAddr),
        ('yield_wait', u32),
    ]
CPSMacAddress = c_void_p32
CPSInAddr = c_long

class CPSSocBuf(Structure):
    _fields_ = [
        ('size', u32),
        ('data', POINTER32(u8)),
    ]
