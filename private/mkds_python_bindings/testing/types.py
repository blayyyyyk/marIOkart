from ctypes import *
from private.mkds_python_bindings.testing.fx import *

PRC_RESAMPLE_METHOD_NONE = 0
PRC_RESAMPLE_METHOD_DISTANCE = 1
PRC_RESAMPLE_METHOD_ANGLE = 2
PRC_RESAMPLE_METHOD_RECURSIVE = 3
PRC_RESAMPLE_METHOD_USER = 256

PRC_RESAMPLE_METHOD_NONE = 0
PRC_RESAMPLE_METHOD_DISTANCE = 1
PRC_RESAMPLE_METHOD_ANGLE = 2
PRC_RESAMPLE_METHOD_RECURSIVE = 3
PRC_RESAMPLE_METHOD_USER = 256

BOOL = c_int
PRCResampleMethod = c_int
REGType16 = c_short
REGType16v = c_short
REGType32 = c_int
REGType32v = c_int
REGType64 = c_long
REGType64v = c_long
REGType8 = c_char
REGType8v = c_char
f32 = c_float
s16 = c_int16
s32 = c_int32
s64 = c_int64
s8 = c_char
u16 = c_short
u32 = c_uint
u64 = c_long
u8 = c_char
vf32 = c_float
vs16 = c_short
vs32 = c_int
vs64 = c_long
vs8 = c_char
vu16 = c_short
vu32 = c_int
vu64 = c_long
vu8 = c_char


class PRCPoint(Structure):
    _fields_ = [
        ('x', s16),
        ('y', s16),
    ]

class PRCBoundingBox(Structure):
    _fields_ = [
        ('x1', s16),
        ('y1', s16),
        ('x2', s16),
        ('y2', s16),
    ]

class PRCStrokes(Structure):
    _fields_ = [
        ('points', u32), #POINTER(PRCPoint)),
        ('size', c_int),
        ('capacity', u32),
    ]

class PRCPrototypeEntry(Structure):
    _fields_ = [
        ('enabled', BOOL),
        ('kind', u32),
        ('code', u16),
        ('correction', fx16),
        ('data', u32),
        ('pointIndex', c_int),
        ('pointCount', u16),
        ('strokeCount', u16),
    ]

class PRCPrototypeList(Structure):
    _fields_ = [
        ('entries', u32), #POINTER(PRCPrototypeEntry)),
        ('entrySize', c_int),
        ('pointArray', u32), #POINTER(PRCPoint)),
        ('pointArraySize', c_int),
    ]

class union__anonymous_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_prc_types_h_170_5_(Union):
    _fields_ = [
        ('normalizeSize', c_int),
        ('regularizeSize', c_int),
    ]
