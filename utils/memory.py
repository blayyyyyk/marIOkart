import struct, math

MAX_S16 = math.pow(16, 3) # 4096
MAX_S32 = math.pow(16, 7) # 268435456

# Memory Addresses
ADDR_ORIENTATION_X = 0x0217B644
ADDR_ORIENTATION_Z = 0x0217B64C
ADDR_POSITION_X = 0x0217B4E0
ADDR_POSITION_Y = 0x0217B4E0 + 0x4
ADDR_POSITION_Z = 0x0217B4E0 + 0x8

def read_s16(emu, addr):
    data = bytes(emu.memory.unsigned[addr:addr+2])
    return struct.unpack("<h", data)[0]
    
def read_s32(emu, addr):
    data = bytes(emu.memory.unsigned[addr:addr+4])
    return struct.unpack("<i", data)[0]
    
def read_cart_angle(emu, addr_x=ADDR_ORIENTATION_X, addr_z=ADDR_ORIENTATION_Z):
    x = read_s16(emu, addr_x) / float(MAX_S16) # normalize to (-1, +1)
    z = read_s16(emu, addr_z) / float(MAX_S16) # normalize to (-1, +1)
    angle = math.atan2(z, x) * 2
    return angle
    
def read_cart_position(emu, addr_x=ADDR_POSITION_X, addr_y=ADDR_POSITION_Y, addr_z=ADDR_POSITION_Z):
    x = read_s32(emu, addr_x) / MAX_S32
    y = read_s32(emu, addr_y) / MAX_S32
    z = read_s32(emu, addr_z) / MAX_S32
    return x, y, z