from ctypes import *
from private.mkds_python_bindings.testing.types import *

WBT_CMD_REQ_NONE = 0
WBT_CMD_REQ_WAIT = 1
WBT_CMD_REQ_SYNC = 2
WBT_CMD_RES_SYNC = 3
WBT_CMD_REQ_GET_BLOCK = 4
WBT_CMD_RES_GET_BLOCK = 5
WBT_CMD_REQ_GET_BLOCKINFO = 6
WBT_CMD_RES_GET_BLOCKINFO = 7
WBT_CMD_REQ_GET_BLOCK_DONE = 8
WBT_CMD_RES_GET_BLOCK_DONE = 9
WBT_CMD_REQ_USER_DATA = 10
WBT_CMD_RES_USER_DATA = 11
WBT_CMD_SYSTEM_CALLBACK = 12
WBT_CMD_PREPARE_SEND_DATA = 13
WBT_CMD_REQ_ERROR = 14
WBT_CMD_RES_ERROR = 15
WBT_CMD_CANCEL = 16

WBT_CMD_REQ_NONE = 0
WBT_CMD_REQ_WAIT = 1
WBT_CMD_REQ_SYNC = 2
WBT_CMD_RES_SYNC = 3
WBT_CMD_REQ_GET_BLOCK = 4
WBT_CMD_RES_GET_BLOCK = 5
WBT_CMD_REQ_GET_BLOCKINFO = 6
WBT_CMD_RES_GET_BLOCKINFO = 7
WBT_CMD_REQ_GET_BLOCK_DONE = 8
WBT_CMD_RES_GET_BLOCK_DONE = 9
WBT_CMD_REQ_USER_DATA = 10
WBT_CMD_RES_USER_DATA = 11
WBT_CMD_SYSTEM_CALLBACK = 12
WBT_CMD_PREPARE_SEND_DATA = 13
WBT_CMD_REQ_ERROR = 14
WBT_CMD_RES_ERROR = 15
WBT_CMD_CANCEL = 16

WBTAidBitmap = c_short
WBTBlockId = c_int
WBTBlockNumEntry = c_short
WBTBlockSeqNo = c_int
WBTBlockSize = c_int
WBTCallback = u32
WBTCommandCounter = c_char
WBTCommandType = c_int
WBTPacketCommand = c_char
WBTPacketSize = c_short
WBTPermission = c_char
WBTResult = c_short


class WBTBlockInfo(Structure):
    _fields_ = [
        ('id', u32),
        ('block_size', s32),
        ('user_id', (u8 * 32)),
    ]

class WBTBlockInfoList(Structure):
    _fields_ = [
        ('data_info', WBTBlockInfo),
        ('next', u32), #POINTER(WBTBlockInfoList)),
        ('data_ptr', u32),
        ('permission_bmp', WBTAidBitmap),
        ('block_type', u16),
    ]

class WBTBlockInfoTable(Structure):
    _fields_ = [
        ('block_info', (POINTER(WBTBlockInfo) * 16)),
    ]

class WBTPacketBitmapTable(Structure):
    _fields_ = [
        ('packet_bitmap', (POINTER(u32) * 16)),
    ]

class WBTRecvBufTable(Structure):
    _fields_ = [
        ('recv_buf', (POINTER(u8) * 16)),
    ]

class WBTRequestSyncCallback(Structure):
    _fields_ = [
        ('num_of_list', WBTBlockNumEntry),
        ('peer_packet_size', s16),
        ('my_packet_size', s16),
        ('pad1', u16),
        ('padd2', (u32 * 2)),
    ]

class WBTGetBlockDoneCallback(Structure):
    _fields_ = [
        ('block_id', u32),
    ]

class WBTPrepareSendDataCallback(Structure):
    _fields_ = [
        ('block_id', u32),
        ('block_seq_no', s32),
        ('data_ptr', u32),
        ('own_packet_size', s16),
        ('padd', u16),
    ]

class WBTRecvUserDataCallback(Structure):
    _fields_ = [
        ('data', (u8 * 9)),
        ('size', u8),
        ('padd', (u8 * 3)),
    ]

class WBTGetBlockCallback(Structure):
    _fields_ = [
        ('block_id', u32),
        ('recv_data_size', u32),
        ('recv_buf_table', WBTRecvBufTable),
        ('pkt_bmp_table', WBTPacketBitmapTable),
    ]

class WBTCommand(Structure):
    _fields_ = [
        ('command', WBTCommandType),
        ('event', WBTCommandType),
        ('target_bmp', u16),
        ('peer_bmp', u16),
        ('my_cmd_counter', WBTCommandCounter),
        ('peer_cmd_counter', WBTCommandCounter),
        ('result', WBTResult),
        ('callback', WBTCallback),
    ]

class union__anonymous_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_wbt_h_307_5_(Union):
    _fields_ = [
        ('sync', WBTRequestSyncCallback),
        ('blockdone', WBTGetBlockDoneCallback),
        ('prepare_send_data', WBTPrepareSendDataCallback),
        ('user_data', WBTRecvUserDataCallback),
        ('get', WBTGetBlockCallback),
    ]
