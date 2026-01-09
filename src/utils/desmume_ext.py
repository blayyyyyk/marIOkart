from ctypes import Structure
import ctypes
from desmume.emulator import (
    DeSmuME as D,
    DeSmuME_Memory as DM,
)
from typing import Type, TypeVar, cast
from typing_extensions import override


class MMUPrefix(ctypes.Structure):
    _fields_ = [
        ("ARM9_ITCM", ctypes.c_uint8 * 0x8000),
        ("ARM9_DTCM", ctypes.c_uint8 * 0x4000),
        ("MAIN_MEM", ctypes.c_uint8 * (16 * 1024 * 1024)),
        # ...
    ]


T = TypeVar('T', bound=Structure)
        

class DeSmuME_Memory(DM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.emu.lib is not None
        self.memoryview = memoryview(MMUPrefix.in_dll(self.emu.lib, "MMU").MAIN_MEM)
        
    def read_struct(self, struct_t: Type[T], addr: int) -> T:
        struct = struct_t.from_buffer(self.memoryview, addr - 0x02000000) # main memory region 
        return struct

class DeSmuME(D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._memory = DeSmuME_Memory(self)
        
    @property
    @override
    def memory(self) -> DeSmuME_Memory:
        return cast(DeSmuME_Memory, self._memory)