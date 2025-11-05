from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.algo_common import *

PRCPrototypeDBParam_Standard = PRCPrototypeDBParam_Common
PRCInputPatternParam_Standard = PRCInputPatternParam_Common
PRCInputPattern_Standard = PRCInputPattern_Common
PRCPrototypeDB_Standard = PRCPrototypeDB_Common
PRCPrototypeDBParam = PRCPrototypeDBParam_Common
PRCInputPatternParam = PRCInputPatternParam_Common
PRCInputPattern = PRCInputPattern_Common
PRCPrototypeDB = PRCPrototypeDB_Common
PRCRecognizeParam = PRCRecognizeParam_Standard

class PRCRecognizeParam_Standard(Structure):
    _fields_ = [
        ('dummy', c_int),
    ]
