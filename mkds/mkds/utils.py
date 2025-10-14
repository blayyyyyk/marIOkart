import struct

def slice_bits(value: int, start: int, end: int) -> int:
    """
    Extracts bits from a 32-bit unsigned integer.

    Args:
        value (int): The 32-bit unsigned integer.
        start (int): The starting bit index (0 = least significant bit).
        end (int): The ending bit index (exclusive).

    Returns:
        int: The extracted bits as a Python integer.
    """
    if not (0 <= start < 32 and 0 < end <= 32 and start < end):
        raise ValueError("start and end must be within 0..32 and start < end")

    # Shift right so 'start' becomes the least significant bit
    shifted = value >> start

    # Mask to keep only (end - start) bits
    mask = (1 << (end - start)) - 1

    return shifted & mask

def read_u8(data: bytes, addr) -> int:
    data = bytes(data[addr : addr + 0x01])
    return struct.unpack("<B", data)[0]


def read_u16(data: bytes, addr) -> int:
    data = bytes(data[addr : addr + 0x02])
    return struct.unpack("<H", data)[0]


def read_u32(data: bytes, addr) -> int:
    data = bytes(data[addr : addr + 0x04])
    return struct.unpack("<I", data)[0]


def read_s8(data: bytes, addr) -> int:
    data = bytes(data[addr : addr + 0x01])
    return struct.unpack("<b", data)[0]

def read_s16(data: bytes, addr) -> int:
    data = bytes(data[addr : addr + 0x02])
    return struct.unpack("<h", data)[0]


def read_s32(data: bytes, addr) -> int:
    data = bytes(data[addr : addr + 0x04])
    return struct.unpack("<i", data)[0]


def read_f16(data: bytes, addr) -> float:
    data = bytes(data[addr : addr + 0x02])
    return struct.unpack("<f", data)[0]


def read_f32(data: bytes, addr) -> float:
    data = bytes(data[addr : addr + 0x04])
    return struct.unpack("<f", data)[0]


def read_fx16(data: bytes, addr) -> float:
    return read_s16(data, addr) / 0x1000  # bit shift 12 bits to the left


def read_fx32(data: bytes, addr) -> float:
    return read_s32(data, addr) / 0x1000  # bit shift 12 bits to the left


def read_vector_2d_fx32(data: bytes, addr, addr2=None) -> tuple[float, float]:
    x = read_fx32(data, addr)
    y = read_fx32(data, addr + 0x04 if addr2 is None else addr2)
    return x, y


def read_vector_3d_fx32(data: bytes, addr, addr2=None, addr3=None) -> tuple[float, float, float]:
    x = read_fx32(data, addr)
    y = read_fx32(data, addr + 0x04 if addr2 is None else addr2)
    z = read_fx32(data, addr + 0x08 if addr3 is None else addr3)
    return x, y, z
    
def read_vector_3d_fx16(data: bytes, addr, addr2=None, addr3=None) -> tuple[float, float, float]:
    x = read_fx16(data, addr)
    y = read_fx16(data, addr + 0x02 if addr2 is None else addr2)
    z = read_fx16(data, addr + 0x04 if addr3 is None else addr3)
    return x, y, z

def read_vector_4d(data: bytes, addr, addr2=None, addr3=None, addr4=None) -> tuple[float, float, float, float]:
    x = read_fx32(data, addr)
    y = read_fx32(data, addr + 0x04 if addr2 is None else addr2)
    z = read_fx32(data, addr + 0x08 if addr3 is None else addr3)
    w = read_fx32(data, addr + 0x0C if addr4 is None else addr4)
    return x, y, z, w

def read_matrix_4d(data: bytes, addr) -> tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float]
]:
    vec_0 = read_vector_4d(data, addr)
    vec_1 = read_vector_4d(data, addr + 0x10)
    vec_2 = read_vector_4d(data, addr + 0x20)
    vec_3 = read_vector_4d(data, addr + 0x30)
    return vec_0, vec_1, vec_2, vec_3

"""
def read_u16_fnl(data: bytes, addr: int) -> tuple[int, bytes]:
    return read_u16(data, addr), data[addr+0x02:]
    
def read_u32_fnl(data: bytes, addr) -> tuple[int, bytes]:
    return read_u32(data, addr), data[addr+0x04:]

def read_s16_fnl(data: bytes, addr) -> tuple[int, bytes]:
    return read_s16(data, addr), data[addr+0x02:]

def read_s32_fnl(data: bytes, addr) -> tuple[int, bytes]:
    return read_s32(data, addr), data[addr+0x04:]
    
def read_f32_fnl(data: bytes, addr) -> tuple[float, bytes]:
    return read_f32(data, addr), data[addr+0x04:]
    
def read_fx16_fnl(data: bytes, addr) -> tuple[float, bytes]:
    return  read_fx16(data, addr), data[addr+0x02:]  # bit shift 12 bits to the left

def read_fx32_fnl(data: bytes, addr) -> tuple[float, bytes]:
    return read_fx32(data, addr), data[addr+0x04:]  # bit shift 12 bits to the left
    
def read_vector(data: bytes, addr: int, *reader_fn) -> list[float | int]:
    if len(reader_fn) == 0:
        return []
    
    val, data = reader_fn[0](data, addr)
    return [val, *read_vector(data, 0, *reader_fn[1:])]
    
def read_vector_2d_fx16(data: bytes, addr: int) -> tuple[float, float]:
    return cast(tuple[float, float], tuple(read_vector(data, addr, read_fx16_fnl, read_fx16_fnl)))
    
def read_vector_2d_fx32(data: bytes, addr: int) -> tuple[float, float]:
    return cast(tuple[float, float], tuple(read_vector(data, addr, read_fx32_fnl, read_fx32_fnl)))
    
def read_vector_3d_fx16(data: bytes, addr: int) -> tuple[float, float, float]:
    return cast(tuple[float, float, float], tuple(read_vector(data, addr, read_fx16_fnl, read_fx16_fnl, read_fx16_fnl)))
    
def read_vector_3d_fx32(data: bytes, addr: int) -> tuple[float, float, float]:
    return cast(tuple[float, float, float], tuple(read_vector(data, addr, read_fx32_fnl, read_fx32_fnl, read_fx32_fnl)))
"""