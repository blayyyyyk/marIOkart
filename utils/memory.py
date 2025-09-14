import struct, math

# Memory Addresses
ADDR_ORIENTATION_X = 0x0217B644
ADDR_ORIENTATION_Z = 0x0217B64C
ADDR_POSITION = 0x0237EA94

RACER_DATA_PTR = 0x0217ACF8
CAMERA_DATA_PTR = 0x0217AA4C

CAMERA_QUATERNION = 0x0217B4E0

def read_u16(data, addr):
    data = bytes(data[addr : addr + 0x02])
    return struct.unpack("<H", data)[0]

# 0x0237EA14

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

def read_vector_4d(data, addr, addr2=None, addr3=None, addr4=None):
    x = read_fx32(data, addr)
    y = read_fx32(data, addr + 0x04 if addr2 is None else addr2)
    z = read_fx32(data, addr + 0x08 if addr3 is None else addr3)
    w = read_fx32(data, addr + 0x0C if addr4 is None else addr4)
    return x, y, z, w
    
def read_matrix_4d(data, addr):
    vec_0 = read_vector_4d(data, addr)
    vec_1 = read_vector_4d(data, addr + 0x10)
    vec_2 = read_vector_4d(data, addr + 0x20)
    vec_3 = read_vector_4d(data, addr + 0x30)
    return vec_0, vec_1, vec_2, vec_3
    
def write_fx32(val: float):
    val *= 4096.0
    return struct.pack("<i", int(val))
    
if __name__ == "__main__":
    x = write_fx32(-3.1643) # \x60\xcd\xff\xff
    x = ''.join(r'\x'+hex(letter)[2:] for letter in x)
    print(x)
    