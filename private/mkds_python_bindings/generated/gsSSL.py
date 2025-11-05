from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.gsCrypt import *
from private.mkds_python_bindings.generated.gsRC4 import *
from private.mkds_python_bindings.generated.gsSHA1 import *
from private.mkds_python_bindings.generated.md5 import *


class gsSSL(Structure):
    _fields_ = [
        ('sessionLen', c_int),
        ('sessionData', (c_char * 255)),
        ('cipherSuite', c_short),
        ('serverpub', gsCryptRSAKey),
        ('sendSeqNBO', (c_char * 8)),
        ('receiveSeqNBO', (c_char * 8)),
        ('clientWriteMACSecret', (c_char * 20)),
        ('clientReadMACSecret', (c_char * 20)),
        ('clientWriteKey', (c_char * 16)),
        ('clientReadKey', (c_char * 16)),
        ('clientWriteIV', (c_char * 16)),
        ('clientReadIV', (c_char * 16)),
        ('clientWriteMACLen', c_int),
        ('clientReadMACLen', c_int),
        ('clientWriteKeyLen', c_int),
        ('clientReadKeyLen', c_int),
        ('clientWriteIVLen', c_int),
        ('clientReadIVLen', c_int),
        ('sendRC4', RC4Context),
        ('recvRC4', RC4Context),
        ('finishHashMD5', MD5_CTX),
        ('finishHashSHA1', SHA1Context),
        ('serverRandom', (c_char * 32)),
        ('clientRandom', (c_char * 32)),
        ('premastersecret', (c_char * 48)),
        ('mastersecret', (c_char * 48)),
    ]

class gsSSLClientHelloMsg(Structure):
    _fields_ = [
        ('header', gsSSLRecordHeaderMsg),
        ('handshakeType', c_char),
        ('lengthNBO', (c_char * 3)),
        ('versionMajor', c_char),
        ('versionMinor', c_char),
        ('time', (c_char * 4)),
        ('random', (c_char * 28)),
        ('sessionIdLen', c_char),
        ('cipherSuitesLength', c_short),
        ('cipherSuites', (c_short * 1)),
        ('compressionMethodLen', c_char),
        ('compressionMethodList', c_char),
    ]

class gsSSLClientKeyExchangeMsg(Structure):
    _fields_ = [
        ('header', gsSSLRecordHeaderMsg),
        ('handshakeType', c_char),
        ('lengthNBO', (c_char * 3)),
    ]

class gsSSLCipherSuiteDesc(Structure):
    _fields_ = [
        ('mSuiteID', c_int),
        ('mKeyLen', c_int),
        ('mMACLen', c_int),
        ('mIVLen', c_int),
    ]

class gsSSLRecordHeaderMsg(Structure):
    _fields_ = [
        ('contentType', c_char),
        ('versionMajor', c_char),
        ('versionMinor', c_char),
        ('lengthNBO', (c_char * 2)),
    ]
