import struct, math

# Memory Addresses
ADDR_ORIENTATION_X = 0x0217B644
ADDR_ORIENTATION_Z = 0x0217B64C
ADDR_POSITION = 0x0217B4E0


def read_u16(data, addr):
    data = bytes(data[addr : addr + 0x02])
    return struct.unpack("<H", data)[0]


def read_u32(data, addr):
    data = bytes(data[addr : addr + 0x04])
    return struct.unpack("<I", data)[0]


def read_s16(data, addr):
    data = bytes(data[addr : addr + 0x02])
    return struct.unpack("<h", data)[0]


def read_s32(data, addr):
    data = bytes(data[addr : addr + 0x04])
    return struct.unpack("<i", data)[0]


def read_fx16(data, addr):
    return read_s16(data, addr) / 4096.0  # bit shift 12 bits to the left


def read_fx32(data, addr):
    return read_s32(data, addr) / 4096.0  # bit shift 12 bits to the left


def read_vector_2d(data, addr, addr2=None):
    x = read_fx32(data, addr)
    y = read_fx32(data, addr + 0x04 if addr2 is None else addr2)
    return x, y


def read_vector_3d(data, addr, addr2=None, addr3=None):
    x = read_fx32(data, addr)
    y = read_fx32(data, addr + 0x04 if addr2 is None else addr2)
    z = read_fx32(data, addr + 0x08 if addr3 is None else addr3)
    return x, y, z
