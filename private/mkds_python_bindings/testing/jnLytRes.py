from ctypes import *
from private.mkds_python_bindings.testing.types import *


class jnui_coord_t(Structure):
    _fields_ = [
        ('coord', s16),
        ('origin', u16),
        ('unk', u16),
    ]

class jnui_bncl_res_header_t(Structure):
    _fields_ = [
        ('magic', u32),
        ('unknown', u16),
        ('nrElements', u16),
    ]

class jnui_bncl_res_element_t(Structure):
    _fields_ = [
        ('x', jnui_coord_t),
        ('y', jnui_coord_t),
        ('cellId', u32),
    ]

class jnui_bncl_res_t(Structure):
    _fields_ = [
        ('header', jnui_bncl_res_header_t),
        ('elements', (jnui_bncl_res_element_t * 0)),
    ]

class jnui_bnll_res_header_t(Structure):
    _fields_ = [
        ('magic', u32),
        ('unknown', u16),
        ('nrElements', u16),
    ]

class jnui_bnll_res_element_t(Structure):
    _fields_ = [
        ('x', jnui_coord_t),
        ('y', jnui_coord_t),
        ('hSpace', s8),
        ('vSpace', s8),
        ('color', u16),
        ('palette', u16),
        ('font', u16),
        ('stringPtr', u32),
    ]

class jnui_bnll_res_t(Structure):
    _fields_ = [
        ('header', jnui_bnll_res_header_t),
        ('elements', (jnui_bnll_res_element_t * 0)),
    ]

class jnui_bnbl_res_header_t(Structure):
    _fields_ = [
        ('magic', u32),
        ('unknown', u16),
        ('nrElements', u16),
    ]

class jnui_bnbl_res_element_t(Structure):
    _fields_ = [
        ('x', jnui_coord_t),
        ('y', jnui_coord_t),
        ('width', u8),
        ('height', u8),
    ]

class jnui_bnbl_res_t(Structure):
    _fields_ = [
        ('header', jnui_bnbl_res_header_t),
        ('elements', (jnui_bnbl_res_element_t * 0)),
    ]
