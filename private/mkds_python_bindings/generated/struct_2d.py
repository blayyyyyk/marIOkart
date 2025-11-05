from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.gxcommon import *
from private.mkds_python_bindings.generated.types import *


class GXScrText32x32(Union):
    _fields_ = [
        ('data16', (u16 * 1024)),
        ('data32', (u32 * 512)),
        ('scr', ((GXScrFmtText * 32) * 32)),
    ]

class GXScrText64x32(Union):
    _fields_ = [
        ('data16', (u16 * 2048)),
        ('data32', (u32 * 1024)),
        ('scr', ((GXScrFmtText * 32) * 64)),
    ]

class GXScrText32x64(Union):
    _fields_ = [
        ('data16', (u16 * 2048)),
        ('data32', (u32 * 1024)),
        ('scr', ((GXScrFmtText * 64) * 32)),
    ]

class GXScrText64x64(Union):
    _fields_ = [
        ('data16', (u16 * 4096)),
        ('data32', (u32 * 2048)),
        ('scr', ((GXScrFmtText * 64) * 64)),
    ]

class GXScrAffine16x16(Union):
    _fields_ = [
        ('data8', (u8 * 256)),
        ('data32', (u32 * 64)),
        ('scr', ((GXScrFmtAffine * 16) * 16)),
    ]

class GXScrAffine32x32(Union):
    _fields_ = [
        ('data8', (u8 * 1024)),
        ('data32', (u32 * 256)),
        ('scr', ((GXScrFmtAffine * 32) * 32)),
    ]

class GXScrAffine64x64(Union):
    _fields_ = [
        ('data8', (u8 * 4096)),
        ('data32', (u32 * 1024)),
        ('scr', ((GXScrFmtAffine * 64) * 64)),
    ]

class GXScrAffine128x128(Union):
    _fields_ = [
        ('data8', (u8 * 16384)),
        ('data32', (u32 * 4096)),
        ('scr', ((GXScrFmtAffine * 128) * 128)),
    ]

class GXScr256Bmp128x128(Union):
    _fields_ = [
        ('data8', (u8 * 16384)),
        ('data32', (u32 * 4096)),
        ('scr', ((GXScrFmt256Bmp * 128) * 128)),
    ]

class GXScr256Bmp256x256(Union):
    _fields_ = [
        ('data8', (u8 * 65536)),
        ('data32', (u32 * 16384)),
        ('scr', ((GXScrFmt256Bmp * 256) * 256)),
    ]

class GXScr256Bmp512x256(Union):
    _fields_ = [
        ('data8', (u8 * 131072)),
        ('data32', (u32 * 32768)),
        ('scr', ((GXScrFmt256Bmp * 256) * 512)),
    ]

class GXScr256Bmp512x512(Union):
    _fields_ = [
        ('data8', (u8 * 262144)),
        ('data32', (u32 * 65536)),
        ('scr', ((GXScrFmt256Bmp * 512) * 512)),
    ]

class GXScrDCBmp128x128(Union):
    _fields_ = [
        ('data16', (u16 * 16384)),
        ('data32', (u32 * 8192)),
        ('scr', ((GXRgba * 128) * 128)),
    ]

class GXScrDCBmp256x256(Union):
    _fields_ = [
        ('data16', (u16 * 65536)),
        ('data32', (u32 * 32768)),
        ('scr', ((GXRgba * 256) * 256)),
    ]

class GXScrDCBmp512x256(Union):
    _fields_ = [
        ('data16', (u16 * 131072)),
        ('data32', (u32 * 65536)),
        ('scr', ((GXRgba * 256) * 512)),
    ]

class GXScrDCBmp512x512(Union):
    _fields_ = [
        ('data16', (u16 * 262144)),
        ('data32', (u32 * 131072)),
        ('scr', ((GXRgba * 512) * 512)),
    ]

class GXCharBGText16(Structure):
    _fields_ = [
        ('ch', (GXCharFmt16 * 1024)),
    ]

class GXCharBGText256(Structure):
    _fields_ = [
        ('ch', (GXCharFmt256 * 1024)),
    ]

class GXCharBGAffine256(Structure):
    _fields_ = [
        ('ch', (GXCharFmt256 * 256)),
    ]

class GXStdPlttData(Structure):
    _fields_ = [
        ('bgPltt', GXBGStdPlttData),
        ('objPltt', GXOBJStdPlttData),
    ]

class GXBGExtPlttData(Structure):
    _fields_ = [
        ('pltt256', (GXBGPltt256 * 16)),
    ]

class GXOBJExtPlttData(Structure):
    _fields_ = [
        ('pltt256', (GXOBJPltt256 * 16)),
    ]
GXScrFmtText = c_short
GXScrFmtAffine = c_char
GXScrFmt256Bmp = c_char

class GXCharFmt16(Union):
    _fields_ = [
        ('data32', (u32 * 8)),
        ('data16', (u16 * 16)),
        ('data8', (u8 * 32)),
    ]

class GXCharFmt256(Union):
    _fields_ = [
        ('data32', (u32 * 16)),
        ('data16', (u16 * 32)),
        ('data8', (u8 * 64)),
    ]

class GXBGStdPlttData(Union):
    _fields_ = [
        ('pltt256', GXBGPltt256),
        ('pltt16', (GXBGPltt16 * 16)),
    ]

class GXOBJStdPlttData(Union):
    _fields_ = [
        ('pltt256', GXOBJPltt256),
        ('pltt16', (GXOBJPltt16 * 16)),
    ]
GXOBJPltt16 = GXBGPltt16
GXOBJPltt256 = GXBGPltt256

class GXBGPltt16(Union):
    _fields_ = [
        ('data16', (u16 * 16)),
        ('data32', (u32 * 8)),
        ('rgb', (GXRgb * 16)),
    ]

class GXBGPltt256(Union):
    _fields_ = [
        ('data16', (u16 * 256)),
        ('data32', (u32 * 128)),
        ('rgb', (GXRgb * 256)),
    ]
