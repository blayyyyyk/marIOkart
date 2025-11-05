from ctypes import *
from private.mkds_python_bindings.testing.algo_common import *

PRCInputPattern = PRCInputPattern_Common
PRCInputPatternParam = PRCInputPatternParam_Common
PRCInputPatternParam_Fine = PRCInputPatternParam_Common
PRCInputPattern_Fine = PRCInputPattern_Common
PRCPrototypeDB = PRCPrototypeDB_Common
PRCPrototypeDBParam = PRCPrototypeDBParam_Common
PRCPrototypeDBParam_Fine = PRCPrototypeDBParam_Common
PRCPrototypeDB_Fine = PRCPrototypeDB_Common
PRCRecognizeParam = PRCRecognizeParam_Fine


class PRCRecognizeParam_Fine(Structure):
    _fields_ = [
        ('lengthFilterThreshold', c_int),
        ('lengthFilterRatio', c_int),
    ]
