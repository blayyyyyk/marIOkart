from ctypes import *
from private.mkds_python_bindings.testing.algo_common import *

PRCInputPattern = PRCInputPattern_Common
PRCInputPatternParam = PRCInputPatternParam_Common
PRCInputPatternParam_Light = PRCInputPatternParam_Common
PRCInputPattern_Light = PRCInputPattern_Common
PRCPrototypeDB = PRCPrototypeDB_Common
PRCPrototypeDBParam = PRCPrototypeDBParam_Common
PRCPrototypeDBParam_Light = PRCPrototypeDBParam_Common
PRCPrototypeDB_Light = PRCPrototypeDB_Common
PRCRecognizeParam = PRCRecognizeParam_Light


class PRCRecognizeParam_Light(Structure):
    _fields_ = [
        ('dummy', c_int),
    ]
