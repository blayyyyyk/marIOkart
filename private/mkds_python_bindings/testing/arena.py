from ctypes import *
from private.mkds_python_bindings.testing.types import *

OS_ARENA_MAIN = 0
OS_ARENA_MAIN_SUBPRIV = 1
OS_ARENA_MAINEX = 2
OS_ARENA_ITCM = 3
OS_ARENA_DTCM = 4
OS_ARENA_SHARED = 5
OS_ARENA_WRAM_MAIN = 6
OS_ARENA_WRAM_SUB = 7
OS_ARENA_WRAM_SUBPRIV = 8
OS_ARENA_MAX = 9

OS_ARENA_MAIN = 0
OS_ARENA_MAIN_SUBPRIV = 1
OS_ARENA_MAINEX = 2
OS_ARENA_ITCM = 3
OS_ARENA_DTCM = 4
OS_ARENA_SHARED = 5
OS_ARENA_WRAM_MAIN = 6
OS_ARENA_WRAM_SUB = 7
OS_ARENA_WRAM_SUBPRIV = 8
OS_ARENA_MAX = 9

OSArenaId = c_int


class OSArenaInfo(Structure):
    _fields_ = [
        ('lo', (u32 * 9)),
        ('hi', (u32 * 9)),
        ('initialized', u16),
        ('padding', (u8 * 2)),
    ]
