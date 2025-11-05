from ctypes import *
from private.mkds_python_bindings.testing.ghostData import *
from private.mkds_python_bindings.testing.inputRecorder import *
from private.mkds_python_bindings.testing.types import *


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_save_saveData_h_7_9_(Structure):
    _fields_ = [
        ('signature', u32),
        ('gap4', (u8 * 8)),
        ('nickname', (u16 * 10)),
        ('unlockBits', (u8 * 4)),
        ('field24', u16),
        ('personalGhostBits', (u8 * 4)),
        ('downloadGhostBits', (u8 * 4)),
        ('nkfeBits', (u8 * 2)),
        ('gap30', u8),
        ('field31', u8),
        ('gap32', u8),
        ('field34', u32),
        ('field38', u32),
        ('field3C', u32),
        ('field40', u32),
        ('field44', u32),
        ('field48', u32),
        ('dwcUserData', c_int),
        ('gap8C', (u8 * 116)),
    ]

class nkpg_t(Structure):
    _fields_ = [
        ('header', ghost_header_t),
        ('inputData', (u8 * 3532)),
    ]

class nkdg_t(Structure):
    _fields_ = [
        ('header', ghost_header_t),
        ('emblem', (u8 * 512)),
        ('inputData', input_rec_recording_t),
        ('padding', u32),
    ]

class save_data_t(Structure):
    _fields_ = [
        ('nksy', u32), #POINTER(nksy_t)),
        ('nkem', u32), #POINTER(u8)),
        ('nkgp', u32),
        ('nkta', u32),
        ('nkmr', u32),
        ('nkpg', u32), #POINTER(nkpg_t)),
        ('nkdg', u32), #POINTER(nkdg_t)),
        ('staffGhost', u32), #POINTER(nkdg_t)),
        ('nkfl', u32),
        ('nkfe', u32),
        ('isBusy', u8),
        ('blockErrorFlags', u16),
        ('error', u32),
        ('field30', (u8 * 4)),
        ('field34', (u8 * 4)),
        ('field38', u8),
        ('field39', u8),
        ('unk3A', (u8 * 2)),
        ('field3C', u8),
        ('unk3D', (u8 * 3)),
        ('field40', u32),
        ('field44', u32),
    ]
