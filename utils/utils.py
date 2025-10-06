import struct

def read_bytes_array(data, addr, max_length):
    arr = bytearray(max_length)
    count = 0
    while count < max_length:
        value = read_u8(data, addr + count)
        arr[count] = value
        count += 1