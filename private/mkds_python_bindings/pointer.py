import ctypes as C
from typing import TypeVar, Generic, overload, Mapping, Any, Self
import sys

c_void_p32 = C.c_uint32

class _CData:
    _b_base_: int
    _b_needsfree_: bool
    _objects: Mapping[Any, int] | None
    def __buffer__(self, flags: int, /) -> memoryview: ...
    def __ctypes_from_outparam__(self, /) -> Self: ...
    if sys.version_info >= (3, 14):
        __pointer_type__: type

T = TypeVar("T", bound=_CData)

def sizeof(obj_or_type) -> int:
    return C.sizeof(obj_or_type)

def addressof(obj) -> int:
    return C.addressof(obj)

def POINTER32(tp: type[T]) -> type[C.c_uint32]:
    """
    Return a 4-byte pointer-like type for use in ctypes Structures that model
    32-bit layouts. Instances support:
      - bool(ptr)               -> False if null
      - int(ptr) / ptr.value    -> 32-bit address
      - ptr.contents            -> tp.from_address(addr)
      - ptr[i]                  -> (tp) at addr + i*sizeof(tp)
      - ptr.cast(U)             -> reinterpret as POINTER32(U)
    Caveat: not a real ctypes pointer; not suitable for argtypes/restype.
    """

    class _Ptr32(C.c_uint32):  # 4 bytes on any host
        _target_ = tp          # for introspection

        def __repr__(self):
            v = int(self.value)
            tgt = getattr(self, "_target_", None)
            tname = getattr(tgt, "__name__", str(tgt))
            return f"<POINTER32[{tname}] 0x{v:08X}>"

        def __bool__(self) -> bool:  # truthiness like a pointer
            return bool(int(self.value))

        def __int__(self) -> int:
            return int(self.value)

        # pointer-like API
        @property
        def contents(self) -> T:
            if not self:
                raise ValueError("NULL POINTER32 has no contents")
            return self._target_.from_address(int(self.value))

        def __getitem__(self, index: int) -> T:
            if index < 0:
                raise IndexError("negative indexing not supported for POINTER32")
            base = int(self.value)
            return self._target_.from_address(base + index * sizeof(self._target_))

        def cast(self, new_tp: type[_CData]):
            """reinterpret as POINTER32(new_tp) without touching memory"""
            New = POINTER32(new_tp)
            out = New()
            out.value = self.value
            return out

    _Ptr32.__name__ = f"POINTER32_{getattr(tp, '__name__', str(tp))}"
    return _Ptr32
