from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.NATify import *


class _InitPacket(Structure):
    _fields_ = [
        ('porttype', c_char),
        ('clientindex', c_char),
        ('usegameport', c_char),
        ('localip', c_int),
        ('localport', c_short),
    ]

class _ReportPacket(Structure):
    _fields_ = [
        ('porttype', c_char),
        ('clientindex', c_char),
        ('negResult', c_char),
        ('natType', NatType),
        ('natMappingScheme', NatMappingScheme),
        ('gamename', (c_char * 50)),
    ]

class _ConnectPacket(Structure):
    _fields_ = [
        ('remoteIP', c_int),
        ('remotePort', c_short),
        ('gotyourdata', c_char),
        ('finished', c_char),
    ]

class _NatNegPacket(Structure):
    _fields_ = [
        ('magic', (c_char * 6)),
        ('version', c_char),
        ('packettype', c_char),
        ('cookie', c_int),
        ('Packet', union__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_dwc_gs_natneg_nninternal_h_118_2),
    ]

class union__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_dwc_gs_natneg_nninternal_h_118_2(Union):
    _fields_ = [
        ('Init', InitPacket),
        ('Connect', ConnectPacket),
        ('Report', ReportPacket),
    ]
