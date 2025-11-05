from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
shaSuccess = 0
shaNull = 1
shaInputTooLong = 2
shaStateError = 3

shaSuccess = 0
shaNull = 1
shaInputTooLong = 2
shaStateError = 3


class SHA1Context(Structure):
    _fields_ = [
        ('Intermediate_Hash', (uint32_t * 5)),
        ('Length_Low', uint32_t),
        ('Length_High', uint32_t),
        ('Message_Block_Index', int_least16_t),
        ('Message_Block', (uint8_t * 64)),
        ('Computed', c_int),
        ('Corrupted', c_int),
    ]
int_least16_t = c_short
uint8_t = c_char
uint32_t = c_int
