from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_jnlib_msg_msg_h_6_9(Structure):
    _fields_ = [
        ('magic1', c_int),
        ('magic2', c_int),
        ('fileSize', c_int),
        ('nrSections', c_int),
        ('field10', c_int),
        ('field14', c_int),
        ('field18', c_int),
        ('field1C', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_jnlib_msg_msg_h_18_9(Structure):
    _fields_ = [
        ('magic', c_int),
        ('size', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_jnlib_msg_msg_h_26_9(Structure):
    _fields_ = [
        ('sectionHeader', jn_msg_bmg_section_header_t),
        ('nrEntries', c_int),
        ('fieldA', c_int),
        ('fieldC', c_int),
        ('offsets', (c_int * 0)),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_jnlib_msg_msg_h_37_9(Structure):
    _fields_ = [
        ('sectionHeader', jn_msg_bmg_section_header_t),
        ('stringData', (c_int * 0)),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_jnlib_msg_msg_h_43_9(Structure):
    _fields_ = [
        ('header', jn_msg_bmg_header_t),
        ('inf1', jn_msg_bmg_inf1_t),
    ]
