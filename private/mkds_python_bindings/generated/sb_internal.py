from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.darray import *
from private.mkds_python_bindings.generated.gsPlatform import *
from private.mkds_python_bindings.generated.gsPlatformSocket import *
from private.mkds_python_bindings.generated.hashtable import *
from private.mkds_python_bindings.generated.sb_crypt import *
from private.mkds_python_bindings.generated.sb_serverbrowsing import *

pi_cryptheader = 0
pi_fixedheader = 1
pi_keylist = 2
pi_uniquevaluelist = 3
pi_servers = 4
pi_finished = 5

pi_cryptheader = 0
pi_fixedheader = 1
pi_keylist = 2
pi_uniquevaluelist = 3
pi_servers = 4
pi_finished = 5

sl_lanbrowse = 0
sl_disconnected = 1
sl_connected = 2
sl_mainlist = 3

sl_lanbrowse = 0
sl_disconnected = 1
sl_connected = 2
sl_mainlist = 3

slc_serveradded = 0
slc_serverupdated = 1
slc_serverdeleted = 2
slc_initiallistcomplete = 3
slc_disconnected = 4
slc_queryerror = 5
slc_publicipdetermined = 6
slc_serverchallengereceived = 7

slc_serveradded = 0
slc_serverupdated = 1
slc_serverdeleted = 2
slc_initiallistcomplete = 3
slc_disconnected = 4
slc_queryerror = 5
slc_publicipdetermined = 6
slc_serverchallengereceived = 7

qe_updatesuccess = 0
qe_updatefailed = 1
qe_engineidle = 2
qe_challengereceived = 3

qe_updatesuccess = 0
qe_updatefailed = 1
qe_engineidle = 2
qe_challengereceived = 3


class _SBKeyValuePair(Structure):
    _fields_ = [
        ('key', POINTER32(c_char)),
        ('value', POINTER32(c_char)),
    ]

class _SBRefString(Structure):
    _fields_ = [
        ('str', POINTER32(c_char)),
        ('refcount', c_int),
    ]

class _KeyInfo(Structure):
    _fields_ = [
        ('keyName', POINTER32(c_char)),
        ('keyType', c_int),
    ]

class _SortInfo(Structure):
    _fields_ = [
        ('sortkey', (c_char * 255)),
        ('comparemode', SBCompareMode),
    ]

class _SBServerList(Structure):
    _fields_ = [
        ('state', SBServerListState),
        ('servers', DArray),
        ('keylist', DArray),
        ('queryforgamename', (c_char * 36)),
        ('queryfromgamename', (c_char * 36)),
        ('queryfromkey', (c_char * 32)),
        ('mychallenge', (c_char * 8)),
        ('inbuffer', POINTER32(c_char)),
        ('inbufferlen', c_int),
        ('popularvalues', (POINTER32(c_char) * 255)),
        ('numpopularvalues', c_int),
        ('expectedelements', c_int),
        ('ListCallback', SBListCallBackFn),
        ('MaploopCallback', SBMaploopCallbackFn),
        ('PlayerSearchCallback', SBPlayerSearchCallbackFn),
        ('instance', c_void_p32),
        ('currsortinfo', SortInfo),
        ('prevsortinfo', SortInfo),
        ('sortascending', SBBool),
        ('mypublicip', goa_uint32),
        ('srcip', goa_uint32),
        ('defaultport', c_short),
        ('lasterror', POINTER32(c_char)),
        ('slsocket', SOCKET),
        ('lanstarttime', gsi_time),
        ('fromgamever', c_int),
        ('cryptkey', GOACryptState),
        ('queryoptions', c_int),
        ('pstate', SBListParseState),
        ('backendgameflags', gsi_u16),
        ('mLanAdapterOverride', POINTER32(c_char)),
        ('deadlist', SBServer),
    ]

class _SBServer(Structure):
    _fields_ = [
        ('publicip', goa_uint32),
        ('publicport', c_short),
        ('privateip', goa_uint32),
        ('privateport', c_short),
        ('icmpip', goa_uint32),
        ('state', c_char),
        ('flags', c_char),
        ('keyvals', HashTable),
        ('updatetime', gsi_time),
        ('querychallenge', gsi_u32),
        ('next', POINTER32(_SBServer)),
        ('splitResponseBitmap', gsi_u8),
    ]

class _SBServerFIFO(Structure):
    _fields_ = [
        ('first', SBServer),
        ('last', SBServer),
        ('count', c_int),
    ]

class _SBQueryEngine(Structure):
    _fields_ = [
        ('queryversion', c_int),
        ('maxupdates', c_int),
        ('querylist', SBServerFIFO),
        ('pendinglist', SBServerFIFO),
        ('querysock', SOCKET),
        ('icmpsock', SOCKET),
        ('mypublicip', goa_uint32),
        ('serverkeys', (c_char * 20)),
        ('numserverkeys', c_int),
        ('ListCallback', SBEngineCallbackFn),
        ('instance', c_void_p32),
    ]

class _ServerBrowser(Structure):
    _fields_ = [
        ('engine', SBQueryEngine),
        ('list', SBServerList),
        ('disconnectFlag', SBBool),
        ('dontUpdate', SBBool),
        ('triggerIP', goa_uint32),
        ('triggerPort', c_short),
        ('BrowserCallback', ServerBrowserCallback),
        ('instance', c_void_p32),
    ]

class _IPHeader(Structure):
    _fields_ = [
        ('ip_hl_ver', gsi_u8),
        ('ip_tos', gsi_u8),
        ('ip_len', gsi_i16),
        ('ip_id', gsi_u16),
        ('ip_off', gsi_i16),
        ('ip_ttl', gsi_u8),
        ('ip_p', gsi_u8),
        ('ip_sum', gsi_u16),
        ('ip_src', SOInAddr),
        ('ip_dst', SOInAddr),
    ]

class _ICMPHeader(Structure):
    _fields_ = [
        ('type', gsi_u8),
        ('code', gsi_u8),
        ('cksum', gsi_u16),
        ('un', union__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_dwc_gs_serverbrowsing_sb_internal_h_349_2),
    ]
SBQueryEnginePtr = c_void_p32
SBServerListPtr = c_void_p32
SBListCallbackReason = c_int
SBQueryEngineCallbackReason = c_int
SBServerListState = c_int
SBListCallBackFn = c_void_p32
SBPlayerSearchCallbackFn = c_void_p32
SBMaploopCallbackFn = c_void_p32
SBListParseState = c_int
SBEngineCallbackFn = c_void_p32

class union__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_dwc_gs_serverbrowsing_sb_internal_h_349_2(Union):
    _fields_ = [
        ('echo', struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_dwc_gs_serverbrowsing_sb_internal_h_350_3),
        ('idseq', gsi_u32),
        ('gateway', gsi_u16),
        ('frag', struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_dwc_gs_serverbrowsing_sb_internal_h_356_3),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_dwc_gs_serverbrowsing_sb_internal_h_350_3(Structure):
    _fields_ = [
        ('id', gsi_u16),
        ('sequence', gsi_u16),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_dwc_gs_serverbrowsing_sb_internal_h_356_3(Structure):
    _fields_ = [
        ('__notused', gsi_u16),
        ('mtu', gsi_u16),
    ]
