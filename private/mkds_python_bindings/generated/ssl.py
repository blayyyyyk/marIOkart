from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.cps import *
from private.mkds_python_bindings.generated.types import *

CPS_CERT_NOERROR = 0
CPS_CERT_NOROOTCA = 1
CPS_CERT_BADSIGNATURE = 2
CPS_CERT_UNKNOWN_SIGALGORITHM = 3
CPS_CERT_UNKNOWN_PUBKEYALGORITHM = 4

CPS_CERT_NOERROR = 0
CPS_CERT_NOROOTCA = 1
CPS_CERT_BADSIGNATURE = 2
CPS_CERT_UNKNOWN_SIGALGORITHM = 3
CPS_CERT_UNKNOWN_PUBKEYALGORITHM = 4

CPS_CERT_NOERROR = 0
CPS_CERT_NOROOTCA = 1
CPS_CERT_BADSIGNATURE = 2
CPS_CERT_UNKNOWN_SIGALGORITHM = 3
CPS_CERT_UNKNOWN_PUBKEYALGORITHM = 4

CPS_CERT_NOERROR = 0
CPS_CERT_NOROOTCA = 1
CPS_CERT_BADSIGNATURE = 2
CPS_CERT_UNKNOWN_SIGALGORITHM = 3
CPS_CERT_UNKNOWN_PUBKEYALGORITHM = 4

CPS_CERT_NOERROR = 0
CPS_CERT_NOROOTCA = 1
CPS_CERT_BADSIGNATURE = 2
CPS_CERT_UNKNOWN_SIGALGORITHM = 3
CPS_CERT_UNKNOWN_PUBKEYALGORITHM = 4


class CPSSslSession(Structure):
    _fields_ = [
        ('sessionID', (u8 * 32)),
        ('master_secret', (u8 * 48)),
        ('when', u32),
        ('ip', CPSInAddr),
        ('port', u16),
        ('valid', u8),
        ('padding', u8),
    ]

class _CPSSslConnection(Structure):
    _fields_ = [
        ('master_secret', (u8 * 48)),
        ('session_cached', u8),
        ('reuse_session', u8),
        ('method', u16),
        ('client_random', (u8 * 32)),
        ('server_random', (u8 * 32)),
        ('common1', union__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitroWiFi_ssl_h_183_5),
        ('send_mac', POINTER32(u8)),
        ('send_key', POINTER32(u8)),
        ('send_iv', POINTER32(u8)),
        ('send_cipher', CPSCipherCtx),
        ('send_seq', (u8 * 8)),
        ('rcv_mac', POINTER32(u8)),
        ('rcv_key', POINTER32(u8)),
        ('rcv_iv', POINTER32(u8)),
        ('rcv_cipher', CPSCipherCtx),
        ('rcv_seq', (u8 * 8)),
        ('sha1_hash', CPSSha1Ctx),
        ('sha1_hash_tmp', CPSSha1Ctx),
        ('md5_hash', CPSMd5Ctx),
        ('md5_hash_tmp', CPSMd5Ctx),
        ('server', u8),
        ('state', u8),
        ('inbuf_decrypted', u8),
        ('padding2', u8),
        ('sig_algorithm', c_int),
        ('pub_algorithm', c_int),
        ('hash_start', POINTER32(u8)),
        ('hash_end', POINTER32(u8)),
        ('hash_val', (u8 * 20)),
        ('hash_len', c_int),
        ('midca_info', CPSCaInfo),
        ('modulus', (u8 * 256)),
        ('modulus_len', u32),
        ('exponent', (u8 * 8)),
        ('exponent_len', c_int),
        ('signature', POINTER32(u8)),
        ('signature_len', c_int),
        ('seen_validity', u8),
        ('seen_pub_algorithm', u8),
        ('seen_attr', u8),
        ('date_ok', u8),
        ('issuer', (c_char * 256)),
        ('subject', (c_char * 256)),
        ('cn', (c_char * 80)),
        ('server_name', POINTER32(c_char)),
        ('cert', POINTER32(u8)),
        ('certlen', c_int),
        ('cur_date', u32),
        ('auth_callback', c_void_p32),
        ('ca_info', POINTER32(POINTER32(CPSCaInfo))),
        ('ca_builtins', c_int),
        ('my_key', POINTER32(CPSPrivateKey)),
        ('my_certificate', POINTER32(CPSCertificate)),
        ('inbuf', POINTER32(u8)),
        ('inbuf_len', c_long),
        ('inbuf_pnt', c_long),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitroWiFi_ssl_h_30_9(Structure):
    _fields_ = [
        ('state', (c_int * 4)),
        ('count', (c_int * 2)),
        ('buffer', (c_int * 64)),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitroWiFi_ssl_h_46_9(Structure):
    _fields_ = [
        ('state', (c_int * 5)),
        ('count', (c_int * 2)),
        ('buffer', (c_int * 64)),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitroWiFi_ssl_h_63_9(Structure):
    _fields_ = [
        ('x', c_int),
        ('y', c_int),
        ('m', (c_int * 256)),
        ('padding', c_int),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitroWiFi_ssl_h_93_9(Structure):
    _fields_ = [
        ('dn', POINTER32(c_char)),
        ('modulus_len', c_int),
        ('modulus', POINTER32(c_int)),
        ('exponent_len', c_int),
        ('exponent', POINTER32(c_int)),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitroWiFi_ssl_h_102_9(Structure):
    _fields_ = [
        ('certificate_len', c_int),
        ('certificate', POINTER32(c_int)),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitroWiFi_ssl_h_108_9(Structure):
    _fields_ = [
        ('modulus_len', c_int),
        ('modulus', POINTER32(c_int)),
        ('prime1_len', c_int),
        ('prime1', POINTER32(c_int)),
        ('prime2_len', c_int),
        ('prime2', POINTER32(c_int)),
        ('exponent1_len', c_int),
        ('exponent1', POINTER32(c_int)),
        ('exponent2_len', c_int),
        ('exponent2', POINTER32(c_int)),
        ('coefficient_len', c_int),
        ('coefficient', POINTER32(c_int)),
    ]

class struct__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitroWiFi_ssl_h_149_9(Structure):
    _fields_ = [
        ('sessionID', (c_int * 32)),
        ('master_secret', (c_int * 48)),
        ('when', c_int),
        ('ip', c_int),
        ('port', c_int),
        ('valid', c_int),
        ('padding', c_int),
    ]

class union__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitroWiFi_ssl_h_160_9(Union):
    _fields_ = [
        ('rc4_ctx', CPSRc4Ctx),
    ]

class CPSCaInfo(Structure):
    _fields_ = [
        ('dn', POINTER32(c_char)),
        ('modulus_len', c_int),
        ('modulus', POINTER32(u8)),
        ('exponent_len', c_int),
        ('exponent', POINTER32(u8)),
    ]

class CPSSha1Ctx(Structure):
    _fields_ = [
        ('state', (u32 * 5)),
        ('count', (u32 * 2)),
        ('buffer', (u8 * 64)),
    ]

class CPSMd5Ctx(Structure):
    _fields_ = [
        ('state', (u32 * 4)),
        ('count', (u32 * 2)),
        ('buffer', (u8 * 64)),
    ]

class CPSPrivateKey(Structure):
    _fields_ = [
        ('modulus_len', c_int),
        ('modulus', POINTER32(u8)),
        ('prime1_len', c_int),
        ('prime1', POINTER32(u8)),
        ('prime2_len', c_int),
        ('prime2', POINTER32(u8)),
        ('exponent1_len', c_int),
        ('exponent1', POINTER32(u8)),
        ('exponent2_len', c_int),
        ('exponent2', POINTER32(u8)),
        ('coefficient_len', c_int),
        ('coefficient', POINTER32(u8)),
    ]

class union__unnamed_at__Users_blakemoody_dev_mariokart_ml_mkds_c_external_include_nitroWiFi_ssl_h_183_5(Union):
    _fields_ = [
        ('sessionID', (u8 * 32)),
        ('key_block', (u8 * 72)),
    ]

class CPSCertificate(Structure):
    _fields_ = [
        ('certificate_len', c_int),
        ('certificate', POINTER32(u8)),
    ]

class CPSCipherCtx(Union):
    _fields_ = [
        ('rc4_ctx', CPSRc4Ctx),
    ]

class CPSRc4Ctx(Structure):
    _fields_ = [
        ('x', u8),
        ('y', u8),
        ('m', (u8 * 256)),
        ('padding', u16),
    ]
