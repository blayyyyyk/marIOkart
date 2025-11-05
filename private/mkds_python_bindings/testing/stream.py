from ctypes import *
MIi_InitReadStreamCallback = u32
MIi_ReadByteStreamCallback = u32
MIi_ReadShortStreamCallback = u32
MIi_ReadWordStreamCallback = u32
MIi_TerminateReadStreamCallback = u32


class MIReadStreamCallbacks(Structure):
    _fields_ = [
        ('initStream', MIi_InitReadStreamCallback),
        ('terminateStream', MIi_TerminateReadStreamCallback),
        ('readByteStream', MIi_ReadByteStreamCallback),
        ('readShortStream', MIi_ReadShortStreamCallback),
        ('readWordStream', MIi_ReadWordStreamCallback),
    ]
