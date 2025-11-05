from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.chat import *
from private.mkds_python_bindings.generated.chatHandlers import *
from private.mkds_python_bindings.generated.chatSocket import *
from private.mkds_python_bindings.generated.darray import *
from private.mkds_python_bindings.generated.hashtable import *

CINoLogin = 0
CIUniqueNickLogin = 1
CIProfileLogin = 2
CIPreAuthLogin = 3

CINoLogin = 0
CIUniqueNickLogin = 1
CIProfileLogin = 2
CIPreAuthLogin = 3

CINoLogin = 0
CIUniqueNickLogin = 1
CIProfileLogin = 2
CIPreAuthLogin = 3

CINoLogin = 0
CIUniqueNickLogin = 1
CIProfileLogin = 2
CIPreAuthLogin = 3


class ciConnection(Structure):
    _fields_ = [
        ('connected', CHATBool),
        ('connecting', CHATBool),
        ('disconnected', CHATBool),
        ('nickErrorCallback', chatNickErrorCallback),
        ('fillInUserCallback', chatFillInUserCallback),
        ('connectCallback', chatConnectCallback),
        ('connectParam', c_void_p32),
        ('chatSocket', ciSocket),
        ('nick', (c_char * 64)),
        ('name', (c_char * 128)),
        ('user', (c_char * 128)),
        ('namespaceID', c_int),
        ('email', (c_char * 64)),
        ('profilenick', (c_char * 32)),
        ('uniquenick', (c_char * 64)),
        ('password', (c_char * 32)),
        ('authtoken', (c_char * 256)),
        ('partnerchallenge', (c_char * 256)),
        ('IP', c_int),
        ('server', (c_char * 128)),
        ('port', c_int),
        ('globalCallbacks', chatGlobalCallbacks),
        ('channelTable', HashTable),
        ('enteringChannelList', DArray),
        ('filterList', POINTER32(ciServerMessageFilter)),
        ('lastFilter', POINTER32(ciServerMessageFilter)),
        ('nextID', c_int),
        ('callbackList', DArray),
        ('quiet', CHATBool),
        ('secretKey', (c_char * 128)),
        ('loginType', CILoginType),
        ('userID', c_int),
        ('profileID', c_int),
    ]
CILoginType = c_int
