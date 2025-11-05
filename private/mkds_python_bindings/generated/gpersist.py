from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
pd_private_ro = 0
pd_private_rw = 1
pd_public_ro = 2
pd_public_rw = 3

pd_private_ro = 0
pd_private_rw = 1
pd_public_ro = 2
pd_public_rw = 3

PersDataCallbackFn = c_void_p32
ProfileCallbackFn = c_void_p32
PersDataSaveCallbackFn = c_void_p32
persisttype_t = c_int
PersAuthCallbackFn = c_void_p32
