from ctypes import *
PRCInputPattern = u32
PRCInputPatternParam = u32
PRCInputPatternParam_Superfine = u32
PRCInputPattern_Superfine = u32
PRCPrototypeDB = u32
PRCPrototypeDBParam = u32
PRCPrototypeDBParam_Superfine = u32
PRCPrototypeDB_Superfine = u32
PRCRecognizeParam = PRCRecognizeParam_Superfine


class PRCRecognizeParam_Superfine(Structure):
    _fields_ = [
        ('lengthFilterThreshold', c_int),
        ('lengthFilterRatio', c_int),
    ]
