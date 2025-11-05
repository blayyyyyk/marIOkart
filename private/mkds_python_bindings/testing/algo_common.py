from ctypes import *
from private.mkds_python_bindings.testing.fx import *
from private.mkds_python_bindings.testing.types import *


class PRCPrototypeDBParam_Common(Structure):
    _fields_ = [
        ('dummy', c_int),
    ]

class PRCInputPatternParam_Common(Structure):
    _fields_ = [
        ('resampleMethod', PRCResampleMethod),
        ('resampleThreshold', c_int),
    ]

class union__anonymous_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_prc_algo_common_h_139_5_(Union):
    _fields_ = [
        ('normalizeSize', c_int),
        ('regularizeSize', c_int),
    ]

class PRCiPatternData_Common(Structure):
    _fields_ = [
        ('strokeCount', u16),
        ('pointCount', u16),
        ('pointArray', u32), #POINTER(PRCPoint)),
        ('strokes', u32), #POINTER(c_int)),
        ('strokeSizes', u32), #POINTER(c_int)),
        ('lineSegmentLengthArray', u32), #POINTER(fx32)),
        ('lineSegmentRatioToStrokeArray', u32), #POINTER(fx16)),
        ('lineSegmentAngleArray', u32), #POINTER(u16)),
        ('strokeLengths', u32), #POINTER(fx32)),
        ('strokeRatios', u32), #POINTER(fx16)),
        ('wholeLength', fx32),
        ('strokeBoundingBoxes', u32), #POINTER(PRCBoundingBox)),
        ('wholeBoundingBox', PRCBoundingBox),
    ]

class PRCiPatternBufferInfo_Common(Structure):
    _fields_ = [
        ('pointArray', u32), #POINTER(PRCPoint)),
        ('strokes', u32), #POINTER(c_int)),
        ('strokeSizes', u32), #POINTER(c_int)),
        ('lineSegmentLengthArray', u32), #POINTER(fx32)),
        ('lineSegmentRatioToStrokeArray', u32), #POINTER(fx16)),
        ('lineSegmentAngleArray', u32), #POINTER(u16)),
        ('strokeLengths', u32), #POINTER(fx32)),
        ('strokeRatios', u32), #POINTER(fx16)),
        ('strokeBoundingBoxes', u32), #POINTER(PRCBoundingBox)),
    ]

class PRCiPrototypeEntry_Common(Structure):
    _fields_ = [
        ('data', PRCiPatternData_Common),
        ('entry', u32), #POINTER(PRCPrototypeEntry)),
    ]

class PRCInputPattern_Common(Structure):
    _fields_ = [
        ('data', PRCiPatternData_Common),
        ('buffer', u32),
        ('bufferSize', u32),
    ]

class union__anonymous_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_prc_algo_common_h_207_5_(Union):
    _fields_ = [
        ('normalizeSize', c_int),
        ('regularizeSize', c_int),
    ]

class PRCiPrototypeListBufferInfo_Common(Structure):
    _fields_ = [
        ('patterns', u32), #POINTER(PRCiPrototypeEntry_Common)),
        ('lineSegmentLengthArray', u32), #POINTER(fx32)),
        ('lineSegmentRatioToStrokeArray', u32), #POINTER(fx16)),
        ('lineSegmentAngleArray', u32), #POINTER(u16)),
        ('strokeArray', u32), #POINTER(c_int)),
        ('strokeSizeArray', u32), #POINTER(c_int)),
        ('strokeLengthArray', u32), #POINTER(fx32)),
        ('strokeRatioArray', u32), #POINTER(fx16)),
        ('strokeBoundingBoxArray', u32), #POINTER(PRCBoundingBox)),
    ]

class PRCPrototypeDB_Common(Structure):
    _fields_ = [
        ('patterns', u32), #POINTER(PRCiPrototypeEntry_Common)),
        ('patternCount', c_int),
        ('lineSegmentLengthArray', u32), #POINTER(fx32)),
        ('lineSegmentRatioToStrokeArray', u32), #POINTER(fx16)),
        ('lineSegmentAngleArray', u32), #POINTER(u16)),
        ('wholePointCount', c_int),
        ('strokeArray', u32), #POINTER(c_int)),
        ('strokeSizeArray', u32), #POINTER(c_int)),
        ('strokeLengthArray', u32), #POINTER(fx32)),
        ('strokeRatioArray', u32), #POINTER(fx16)),
        ('strokeBoundingBoxArray', u32), #POINTER(PRCBoundingBox)),
        ('wholeStrokeCount', c_int),
        ('maxPointCount', c_int),
        ('maxStrokeCount', c_int),
        ('prototypeList', u32), #POINTER(PRCPrototypeList)),
        ('buffer', u32),
        ('bufferSize', u32),
    ]

class union__anonymous_at__Users_blakemoody_dev_mariokart_ml_mkds_c_src_stub_includes_nitro_prc_algo_common_h_262_5_(Union):
    _fields_ = [
        ('normalizeSize', c_int),
        ('regularizeSize', c_int),
    ]
