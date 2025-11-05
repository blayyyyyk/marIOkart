from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
SBFalse = 0
SBTrue = 1

SBFalse = 0
SBTrue = 1

sbe_noerror = 0
sbe_socketerror = 1
sbe_dnserror = 2
sbe_connecterror = 3
sbe_dataerror = 4
sbe_allocerror = 5
sbe_paramerror = 6
sbe_duplicateupdateerror = 7

sbe_noerror = 0
sbe_socketerror = 1
sbe_dnserror = 2
sbe_connecterror = 3
sbe_dataerror = 4
sbe_allocerror = 5
sbe_paramerror = 6
sbe_duplicateupdateerror = 7

sb_disconnected = 0
sb_listxfer = 1
sb_querying = 2
sb_connected = 3

sb_disconnected = 0
sb_listxfer = 1
sb_querying = 2
sb_connected = 3

sbc_serveradded = 0
sbc_serverupdated = 1
sbc_serverupdatefailed = 2
sbc_serverdeleted = 3
sbc_updatecomplete = 4
sbc_queryerror = 5
sbc_serverchallengereceived = 6

sbc_serveradded = 0
sbc_serverupdated = 1
sbc_serverupdatefailed = 2
sbc_serverdeleted = 3
sbc_updatecomplete = 4
sbc_queryerror = 5
sbc_serverchallengereceived = 6

sbcm_int = 0
sbcm_float = 1
sbcm_strcase = 2
sbcm_stricase = 3

sbcm_int = 0
sbcm_float = 1
sbcm_strcase = 2
sbcm_stricase = 3

SBFalse = 0
SBTrue = 1

SBFalse = 0
SBTrue = 1

sbe_noerror = 0
sbe_socketerror = 1
sbe_dnserror = 2
sbe_connecterror = 3
sbe_dataerror = 4
sbe_allocerror = 5
sbe_paramerror = 6
sbe_duplicateupdateerror = 7

sbe_noerror = 0
sbe_socketerror = 1
sbe_dnserror = 2
sbe_connecterror = 3
sbe_dataerror = 4
sbe_allocerror = 5
sbe_paramerror = 6
sbe_duplicateupdateerror = 7

sb_disconnected = 0
sb_listxfer = 1
sb_querying = 2
sb_connected = 3

sb_disconnected = 0
sb_listxfer = 1
sb_querying = 2
sb_connected = 3

sbc_serveradded = 0
sbc_serverupdated = 1
sbc_serverupdatefailed = 2
sbc_serverdeleted = 3
sbc_updatecomplete = 4
sbc_queryerror = 5
sbc_serverchallengereceived = 6

sbc_serveradded = 0
sbc_serverupdated = 1
sbc_serverupdatefailed = 2
sbc_serverdeleted = 3
sbc_updatecomplete = 4
sbc_queryerror = 5
sbc_serverchallengereceived = 6

sbcm_int = 0
sbcm_float = 1
sbcm_strcase = 2
sbcm_stricase = 3

sbcm_int = 0
sbcm_float = 1
sbcm_strcase = 2
sbcm_stricase = 3

SBError = c_int
SBCallbackReason = c_int
SBState = c_int
ServerBrowser = c_void_p32
SBServerKeyEnumFn = c_void_p32
SBCompareMode = c_int
SBServer = c_void_p32
SBBool = c_int
ServerBrowserCallback = c_void_p32
