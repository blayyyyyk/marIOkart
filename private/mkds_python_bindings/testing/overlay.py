from ctypes import *
from private.mkds_python_bindings.testing.exMemory import *
from private.mkds_python_bindings.testing.file import *
from private.mkds_python_bindings.testing.nnsfnd import *
from private.mkds_python_bindings.testing.rom import *
from private.mkds_python_bindings.testing.types import *

FSOverlayID = c_int
FSOverlayInitFunc = u32


class overlay_data_overlayinfo_t(Structure):
    _fields_ = [
        ('id', u32),
        ('start', u32),
        ('end', u32),
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

class FSOverlayInfoHeader(Structure):
    _fields_ = [
        ('id', u32),
        ('ram_address', u32), #POINTER(u8)),
        ('ram_size', u32),
        ('bss_size', u32),
        ('sinit_init', u32), #POINTER(FSOverlayInitFunc)),
        ('sinit_init_end', u32), #POINTER(FSOverlayInitFunc)),
        ('file_id', u32),
        ('compressed', u32),
        ('flag', u32),
    ]

class FSOverlayInfo(Structure):
    _fields_ = [
        ('header', FSOverlayInfoHeader),
        ('target', MIProcessor),
        ('file_pos', CARDRomRegion),
    ]
