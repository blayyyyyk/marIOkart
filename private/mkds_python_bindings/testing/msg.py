from ctypes import *
from private.mkds_python_bindings.testing.types import *


class jn_msg_bmg_header_t(Structure):
    _fields_ = [
        ('magic1', u32),
        ('magic2', u32),
        ('fileSize', u32),
        ('nrSections', u32),
        ('field10', u32),
        ('field14', u32),
        ('field18', u32),
        ('field1C', u32),
    ]

class jn_msg_bmg_section_header_t(Structure):
    _fields_ = [
        ('magic', u32),
        ('size', u32),
    ]

class jn_msg_bmg_inf1_t(Structure):
    _fields_ = [
        ('sectionHeader', jn_msg_bmg_section_header_t),
        ('nrEntries', u16),
        ('fieldA', u16),
        ('fieldC', u32),
        ('offsets', (u32 * 0)),
    ]

class jn_msg_bmg_dat1_t(Structure):
    _fields_ = [
        ('sectionHeader', jn_msg_bmg_section_header_t),
        ('stringData', (u16 * 0)),
    ]

class jn_msg_bmg_t(Structure):
    _fields_ = [
        ('header', jn_msg_bmg_header_t),
        ('inf1', jn_msg_bmg_inf1_t),
    ]
