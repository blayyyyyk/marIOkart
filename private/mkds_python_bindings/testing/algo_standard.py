from ctypes import *
from private.mkds_python_bindings.testing.algo_common import *

PRCInputPattern = PRCInputPattern_Common
PRCInputPatternParam = PRCInputPatternParam_Common
PRCInputPatternParam_Standard = PRCInputPatternParam_Common
PRCInputPattern_Standard = PRCInputPattern_Common
PRCPrototypeDB = PRCPrototypeDB_Common
PRCPrototypeDBParam = PRCPrototypeDBParam_Common
PRCPrototypeDBParam_Standard = PRCPrototypeDBParam_Common
PRCPrototypeDB_Standard = PRCPrototypeDB_Common
PRCRecognizeParam = PRCRecognizeParam_Standard


class PRCRecognizeParam_Standard(Structure):
    _fields_ = [
        ('dummy', c_int),
    ]
