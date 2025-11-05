from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.ghostData import *
from private.mkds_python_bindings.generated.inputRecorder import *


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_save_saveData_h_8_9(Structure):
    _fields_ = [
        ('signature', c_int),
        ('gap4', (c_int * 8)),
        ('nickname', (c_int * 10)),
        ('unlockBits', (c_int * 4)),
        ('field24', c_int),
        ('personalGhostBits', (c_int * 4)),
        ('downloadGhostBits', (c_int * 4)),
        ('nkfeBits', (c_int * 2)),
        ('gap30', c_int),
        ('field31', c_int),
        ('gap32', c_int),
        ('field34', c_int),
        ('field38', c_int),
        ('field3C', c_int),
        ('field40', c_int),
        ('field44', c_int),
        ('field48', c_int),
        ('dwcUserData', c_int),
        ('gap8C', (c_int * 116)),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_save_saveData_h_31_9(Structure):
    _fields_ = [
        ('header', ghost_header_t),
        ('inputData', (c_int * 3532)),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_save_saveData_h_37_9(Structure):
    _fields_ = [
        ('header', ghost_header_t),
        ('emblem', (c_int * 512)),
        ('inputData', input_rec_recording_t),
        ('padding', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_save_saveData_h_45_9(Structure):
    _fields_ = [
        ('nksy', POINTER32(nksy_t)),
        ('nkem', POINTER32(c_int)),
        ('nkgp', c_int),
        ('nkta', c_int),
        ('nkmr', c_int),
        ('nkpg', POINTER32(nkpg_t)),
        ('nkdg', POINTER32(nkdg_t)),
        ('staffGhost', POINTER32(nkdg_t)),
        ('nkfl', c_int),
        ('nkfe', c_int),
        ('isBusy', c_int),
        ('blockErrorFlags', c_int),
        ('error', c_int),
        ('field30', (c_int * 4)),
        ('field34', (c_int * 4)),
        ('field38', c_int),
        ('field39', c_int),
        ('unk3A', (c_int * 2)),
        ('field3C', c_int),
        ('unk3D', (c_int * 3)),
        ('field40', c_int),
        ('field44', c_int),
    ]
