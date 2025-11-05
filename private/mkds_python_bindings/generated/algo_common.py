from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.types import *


class union__anonymous_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_prc_algo_common_h_139_5(Union):
    _fields_ = [
        ('normalizeSize', c_int),
        ('regularizeSize', c_int),
    ]

class PRCiPatternBufferInfo_Common(Structure):
    _fields_ = [
        ('pointArray', POINTER32(PRCPoint)),
        ('strokes', POINTER32(c_int)),
        ('strokeSizes', POINTER32(c_int)),
        ('lineSegmentLengthArray', POINTER32(c_int)),
        ('lineSegmentRatioToStrokeArray', POINTER32(c_int)),
        ('lineSegmentAngleArray', POINTER32(u16)),
        ('strokeLengths', POINTER32(c_int)),
        ('strokeRatios', POINTER32(c_int)),
        ('strokeBoundingBoxes', POINTER32(PRCBoundingBox)),
    ]

class union__anonymous_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_prc_algo_common_h_207_5(Union):
    _fields_ = [
        ('normalizeSize', c_int),
        ('regularizeSize', c_int),
    ]

class PRCiPrototypeListBufferInfo_Common(Structure):
    _fields_ = [
        ('patterns', POINTER32(PRCiPrototypeEntry_Common)),
        ('lineSegmentLengthArray', POINTER32(c_int)),
        ('lineSegmentRatioToStrokeArray', POINTER32(c_int)),
        ('lineSegmentAngleArray', POINTER32(u16)),
        ('strokeArray', POINTER32(c_int)),
        ('strokeSizeArray', POINTER32(c_int)),
        ('strokeLengthArray', POINTER32(c_int)),
        ('strokeRatioArray', POINTER32(c_int)),
        ('strokeBoundingBoxArray', POINTER32(PRCBoundingBox)),
    ]

class union__anonymous_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitro_prc_algo_common_h_262_5(Union):
    _fields_ = [
        ('normalizeSize', c_int),
        ('regularizeSize', c_int),
    ]

class PRCPrototypeDBParam_Common(Structure):
    _fields_ = [
        ('dummy', c_int),
    ]

class PRCInputPatternParam_Common(Structure):
    _fields_ = [
        ('resampleMethod', PRCResampleMethod),
        ('resampleThreshold', c_int),
    ]

class PRCInputPattern_Common(Structure):
    _fields_ = [
        ('data', PRCiPatternData_Common),
        ('buffer', c_void_p32),
        ('bufferSize', u32),
    ]

class PRCPrototypeDB_Common(Structure):
    _fields_ = [
        ('patterns', POINTER32(PRCiPrototypeEntry_Common)),
        ('patternCount', c_int),
        ('lineSegmentLengthArray', POINTER32(c_int)),
        ('lineSegmentRatioToStrokeArray', POINTER32(c_int)),
        ('lineSegmentAngleArray', POINTER32(u16)),
        ('wholePointCount', c_int),
        ('strokeArray', POINTER32(c_int)),
        ('strokeSizeArray', POINTER32(c_int)),
        ('strokeLengthArray', POINTER32(c_int)),
        ('strokeRatioArray', POINTER32(c_int)),
        ('strokeBoundingBoxArray', POINTER32(PRCBoundingBox)),
        ('wholeStrokeCount', c_int),
        ('maxPointCount', c_int),
        ('maxStrokeCount', c_int),
        ('prototypeList', POINTER32(PRCPrototypeList)),
        ('buffer', c_void_p32),
        ('bufferSize', u32),
    ]

class PRCiPrototypeEntry_Common(Structure):
    _fields_ = [
        ('data', PRCiPatternData_Common),
        ('entry', POINTER32(PRCPrototypeEntry)),
    ]

class PRCiPatternData_Common(Structure):
    _fields_ = [
        ('strokeCount', u16),
        ('pointCount', u16),
        ('pointArray', POINTER32(PRCPoint)),
        ('strokes', POINTER32(c_int)),
        ('strokeSizes', POINTER32(c_int)),
        ('lineSegmentLengthArray', POINTER32(c_int)),
        ('lineSegmentRatioToStrokeArray', POINTER32(c_int)),
        ('lineSegmentAngleArray', POINTER32(u16)),
        ('strokeLengths', POINTER32(c_int)),
        ('strokeRatios', POINTER32(c_int)),
        ('wholeLength', c_int),
        ('strokeBoundingBoxes', POINTER32(PRCBoundingBox)),
        ('wholeBoundingBox', PRCBoundingBox),
    ]
