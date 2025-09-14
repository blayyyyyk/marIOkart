"""Reads the Y9 overlay table from a ROM file.

Imports:
    struct

Usage:
    python read_y9.py

"""

import struct

with open("rom_unpacked/y9.bin", "rb") as f:
    data = f.read()

for addr in range(0x00, len(data), 0x20):
    data_bytes = bytes(data[addr: addr + 0x20])
    entry = struct.unpack("<iiiiiiii", data_bytes)
    
    print("\n\n"+"-"*10+"Overlay Entry"+"-"*10)
    print(f"Overlay Number: {entry[0]}")
    print(f"RAM Address: {hex(entry[1])}")
    print(f"RAM Size: {entry[2]} ({hex(entry[2])}) bytes")
    print(f"BSS Size: {entry[3]} ({hex(entry[3])}) bytes")
    print(f"Start Address: {hex(entry[4])}")
    print(f"End Address: {hex(entry[5])}")
    print(f"File ID: {entry[6]}")
    print(f"Reserved: {entry[7]}")
