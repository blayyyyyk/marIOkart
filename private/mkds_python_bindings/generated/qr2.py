from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.gsPlatform import *
from private.mkds_python_bindings.generated.gsPlatformSocket import *

e_qrnoerror = 0
e_qrwsockerror = 1
e_qrbinderror = 2
e_qrdnserror = 3
e_qrconnerror = 4
e_qrnochallengeerror = 5
qr2_error_t_count = 6

e_qrnoerror = 0
e_qrwsockerror = 1
e_qrbinderror = 2
e_qrdnserror = 3
e_qrconnerror = 4
e_qrnochallengeerror = 5
qr2_error_t_count = 6

key_server = 0
key_player = 1
key_team = 2
key_type_count = 3

key_server = 0
key_player = 1
key_team = 2
key_type_count = 3


class qr2_implementation_s(Structure):
    _fields_ = [
        ('hbsock', SOCKET),
        ('gamename', (c_char * 64)),
        ('secret_key', (c_char * 64)),
        ('instance_key', (c_char * 4)),
        ('server_key_callback', qr2_serverkeycallback_t),
        ('player_key_callback', qr2_playerteamkeycallback_t),
        ('team_key_callback', qr2_playerteamkeycallback_t),
        ('key_list_callback', qr2_keylistcallback_t),
        ('playerteam_count_callback', qr2_countcallback_t),
        ('adderror_callback', qr2_adderrorcallback_t),
        ('nn_callback', qr2_natnegcallback_t),
        ('cm_callback', qr2_clientmessagecallback_t),
        ('pa_callback', qr2_publicaddresscallback_t),
        ('lastheartbeat', gsi_time),
        ('lastka', gsi_time),
        ('userstatechangerequested', c_int),
        ('listed_state', c_int),
        ('ispublic', c_int),
        ('qport', c_int),
        ('read_socket', c_int),
        ('nat_negotiate', c_int),
        ('hbaddr', SOSockAddrIn),
        ('cdkeyprocess', cdkey_process_t),
        ('client_message_keys', (c_int * 10)),
        ('cur_message_key', c_int),
        ('publicip', c_int),
        ('publicport', c_short),
        ('udata', c_void_p32),
        ('backendoptions', gsi_u8),
        ('ipverify', (qr2_ipverify_info_s * 200)),
    ]
qr2_buffer_t = c_void_p32
qr2_error_t = c_int
qr2_key_type = c_int
qr2_keybuffer_t = c_void_p32
qr2_t = c_void_p32
qr2_ipverify_node_t = c_void_p32
qr2_publicaddresscallback_t = c_void_p32
qr2_countcallback_t = c_void_p32
qr2_playerteamkeycallback_t = c_void_p32
qr2_adderrorcallback_t = c_void_p32
qr2_serverkeycallback_t = c_void_p32
qr2_keylistcallback_t = c_void_p32
cdkey_process_t = c_void_p32
qr2_natnegcallback_t = c_void_p32

class qr2_ipverify_info_s(Structure):
    _fields_ = [
        ('addr', SOSockAddrIn),
        ('challenge', gsi_u32),
        ('createtime', gsi_time),
    ]
qr2_clientmessagecallback_t = c_void_p32
