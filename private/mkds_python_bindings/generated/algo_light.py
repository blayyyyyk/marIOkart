from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.algo_common import *

PRCPrototypeDBParam = PRCPrototypeDBParam_Common
PRCInputPatternParam = PRCInputPatternParam_Common
PRCInputPattern = PRCInputPattern_Common
PRCPrototypeDB = PRCPrototypeDB_Common
PRCRecognizeParam = PRCRecognizeParam_Light
PRCPrototypeDBParam_Light = PRCPrototypeDBParam_Common
PRCInputPatternParam_Light = PRCInputPatternParam_Common
PRCInputPattern_Light = PRCInputPattern_Common
PRCPrototypeDB_Light = PRCPrototypeDB_Common

class PRCRecognizeParam_Light(Structure):
    _fields_ = [
        ('dummy', c_int),
    ]
