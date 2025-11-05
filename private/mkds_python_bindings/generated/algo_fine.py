from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.algo_common import *

PRCPrototypeDBParam = PRCPrototypeDBParam_Common
PRCInputPatternParam = PRCInputPatternParam_Common
PRCInputPattern = PRCInputPattern_Common
PRCPrototypeDB = PRCPrototypeDB_Common
PRCRecognizeParam = PRCRecognizeParam_Fine
PRCPrototypeDBParam_Fine = PRCPrototypeDBParam_Common
PRCInputPatternParam_Fine = PRCInputPatternParam_Common
PRCInputPattern_Fine = PRCInputPattern_Common
PRCPrototypeDB_Fine = PRCPrototypeDB_Common

class PRCRecognizeParam_Fine(Structure):
    _fields_ = [
        ('lengthFilterThreshold', c_int),
        ('lengthFilterRatio', c_int),
    ]
