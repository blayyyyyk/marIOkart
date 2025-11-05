from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_jnlib_ui_jnLytRes_h_11_9(Structure):
    _fields_ = [
        ('coord', c_int),
        ('origin', c_int),
        ('unk', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_jnlib_ui_jnLytRes_h_20_9(Structure):
    _fields_ = [
        ('magic', c_int),
        ('unknown', c_int),
        ('nrElements', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_jnlib_ui_jnLytRes_h_27_9(Structure):
    _fields_ = [
        ('x', jnui_coord_t),
        ('y', jnui_coord_t),
        ('cellId', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_jnlib_ui_jnLytRes_h_34_9(Structure):
    _fields_ = [
        ('header', jnui_bncl_res_header_t),
        ('elements', (jnui_bncl_res_element_t * 0)),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_jnlib_ui_jnLytRes_h_40_9(Structure):
    _fields_ = [
        ('magic', c_int),
        ('unknown', c_int),
        ('nrElements', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_jnlib_ui_jnLytRes_h_47_9(Structure):
    _fields_ = [
        ('x', jnui_coord_t),
        ('y', jnui_coord_t),
        ('hSpace', c_int),
        ('vSpace', c_int),
        ('color', c_int),
        ('palette', c_int),
        ('font', c_int),
        ('stringPtr', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_jnlib_ui_jnLytRes_h_59_9(Structure):
    _fields_ = [
        ('header', jnui_bnll_res_header_t),
        ('elements', (jnui_bnll_res_element_t * 0)),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_jnlib_ui_jnLytRes_h_65_9(Structure):
    _fields_ = [
        ('magic', c_int),
        ('unknown', c_int),
        ('nrElements', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_jnlib_ui_jnLytRes_h_72_9(Structure):
    _fields_ = [
        ('x', jnui_coord_t),
        ('y', jnui_coord_t),
        ('width', c_int),
        ('height', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_jnlib_ui_jnLytRes_h_82_9(Structure):
    _fields_ = [
        ('header', jnui_bnbl_res_header_t),
        ('elements', (jnui_bnbl_res_element_t * 0)),
    ]
