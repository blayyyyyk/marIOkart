import struct, math

MAX_S16 = math.pow(16, 3) # 4096
MAX_S32 = math.pow(16, 7) # 268435456

# Memory Addresses
ADDR_ORIENTATION_X = 0x0217B644
ADDR_ORIENTATION_Z = 0x0217B64C
ADDR_POSITION_X = 0x0217B4E0
ADDR_POSITION_Y = 0x0217B4E0 + 0x4
ADDR_POSITION_Z = 0x0217B4E0 + 0x8

def read_f32(data, addr):
    data = bytes(data[addr:addr+0x04])
    return struct.unpack("<f", data)[0]

def read_u16(data, addr):
    data = bytes(data[addr:addr+0x02])
    return struct.unpack("<H", data)[0]
    
def read_u32(data, addr):
    data = bytes(data[addr:addr+0x04])
    return struct.unpack("<I", data)[0]

def read_s16(data, addr):
    data = bytes(data[addr:addr+0x02])
    return struct.unpack("<h", data)[0]
    
def read_s32(data, addr):
    data = bytes(data[addr:addr+0x04])
    return struct.unpack("<i", data)[0]
    
def read_fx16(data, addr):
    return read_s16(data, addr) / 4096.0

def read_fx32(data, addr):
    return read_s32(data, addr) / 4096.0
    
    
def read_vector_2d(data, addr):
    x = read_fx32(data, addr)
    y = read_fx32(data, addr+0x04)
    return x, y
    
def read_vector_3d(data, addr):
    x = read_fx32(data, addr)
    y = read_fx32(data, addr+0x04)
    z = read_fx32(data, addr+0x08)
    return x, y, z
    
def emu_read_s16(emu, addr):
    return read_fx16(emu.memory.unsigned, addr)
    
def emu_read_s32(emu, addr):
    return read_fx32(emu.memory.unsigned, addr)
    
def read_cart_angle(emu, addr_x=ADDR_ORIENTATION_X, addr_z=ADDR_ORIENTATION_Z):
    x = emu_read_s16(emu, addr_x) # normalized to (-1, +1)
    z = emu_read_s16(emu, addr_z) # normalized to (-1, +1)
    angle = math.atan2(z, x) * 2
    return angle
    
def read_cart_position(emu, addr_x=ADDR_POSITION_X, addr_y=ADDR_POSITION_Y, addr_z=ADDR_POSITION_Z):
    x = emu_read_s32(emu, addr_x)
    y = emu_read_s32(emu, addr_y)
    z = emu_read_s32(emu, addr_z)
    return x, y, z