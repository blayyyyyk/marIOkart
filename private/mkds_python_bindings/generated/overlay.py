from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.exMemory import *
from private.mkds_python_bindings.generated.file import *
from private.mkds_python_bindings.generated.heapcommon import *
from private.mkds_python_bindings.generated.rom import *
from private.mkds_python_bindings.generated.types import *


class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_overlay_h_31_9(Structure):
    _fields_ = [
        ('id', c_int),
        ('start', c_int),
        ('end', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_overlay_h_38_9(Structure):
    _fields_ = [
        ('curOverlay', c_int),
        ('state', c_int),
        ('overlays', (overlay_data_overlayinfo_t * 3)),
        ('overlayInfo', c_int),
        ('overlayFile', c_int),
        ('overlayFrmHeap', c_int),
        ('overlayExpHeap', c_int),
        ('overlayRegionStart', c_int),
        ('overlayRegionEnd', c_int),
    ]

class overlay_data_t(Structure):
    _fields_ = [
        ('curOverlay', u32),
        ('state', u32),
        ('overlays', (overlay_data_overlayinfo_t * 3)),
        ('overlayInfo', FSOverlayInfo),
        ('overlayFile', FSFile),
        ('overlayFrmHeap', NNSFndHeapHandle),
        ('overlayExpHeap', NNSFndHeapHandle),
        ('overlayRegionStart', u32),
        ('overlayRegionEnd', u32),
    ]
FSOverlayID = c_long

class FSOverlayInfo(Structure):
    _fields_ = [
        ('header', FSOverlayInfoHeader),
        ('target', MIProcessor),
        ('file_pos', CARDRomRegion),
    ]

class overlay_data_overlayinfo_t(Structure):
    _fields_ = [
        ('id', u32),
        ('start', u32),
        ('end', u32),
    ]

class FSOverlayInfoHeader(Structure):
    _fields_ = [
        ('id', u32),
        ('ram_address', POINTER32(u8)),
        ('ram_size', u32),
        ('bss_size', u32),
        ('sinit_init', POINTER32(FSOverlayInitFunc)),
        ('sinit_init_end', POINTER32(FSOverlayInitFunc)),
        ('file_id', u32),
        ('compressed', u32),
        ('flag', u32),
    ]
FSOverlayInitFunc = c_void_p32
