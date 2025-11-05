from ctypes import *
from private.mkds_python_bindings.testing.exMemory import *
from private.mkds_python_bindings.testing.system import *
from private.mkds_python_bindings.testing.types import *


class CTRDGHeader(Structure):
    _fields_ = [
        ('startAddress', u32),
        ('nintendoLogo', (u8 * 156)),
        ('titleName', (c_char * 12)),
        ('gameCode', u32),
        ('makerCode', u16),
        ('isRomCode', u8),
        ('machineCode', u8),
        ('deviceType', u8),
        ('exLsiID', (u8 * 3)),
        ('reserved_A', (u8 * 4)),
        ('softVersion', u8),
        ('complement', u8),
        ('moduleID', u16),
    ]

class CTRDGModuleID(Structure):
    _fields_ = [
    ]

class union__anonymous_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_ctrdg_common_ctrdg_common_h_149_5_(Union):
    _fields_ = [
        ('raw', u16),
    ]

class struct__anonymous_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_ctrdg_common_ctrdg_common_h_151_9_(Structure):
    _fields_ = [
        ('bitID', u8),
        ('numberID', u8),
        ('_', u8),
        ('disableExLsiID', u8),
    ]

class CTRDGModuleInfo(Structure):
    _fields_ = [
        ('moduleID', CTRDGModuleID),
        ('exLsiID', (u8 * 3)),
        ('isAgbCartridge', u8),
        ('detectPullOut', u8),
        ('_', u8),
        ('makerCode', u16),
        ('gameCode', u32),
    ]

class CTRDGRomCycle(Structure):
    _fields_ = [
        ('c1', MICartridgeRomCycle1st),
        ('c2', MICartridgeRomCycle2nd),
    ]

class CTRDGLockByProc(Structure):
    _fields_ = [
        ('locked', BOOL),
        ('irq', OSIntrMode),
    ]
