from __future__ import annotations

import json
import struct
import sys

# --- Encryption / Decryption ------------------------------------------
mask1 = [
	0xC0A2CFB9, 0xFCA4F59A, 0xD87E2588, 0xB615DD30,
	0xEACC91EE, 0xB6A8F95D, 0x7BD8B080, 0xCEC11555,
	0x59CEF8CE, 0xFFF88DB3, 0xA867F2A3,
]

mask2 = [
	0x54E85EA2, 0xDE7CE656, 0xD25261B4, 0x401D1EC4,
	0x220DECE2, 0x1F0CF6F4, 0x7176B2A6, 0x58D602EC,
	0x859D5B49, 0xE5258B0C, 0xB31A2910, 0x2DDE04B1,
	0xEEEFA031, 0x5B852490, 0x44AD6965, 0x8E3206A5,
	0x5E22B868,
]

mask3 = [
	0x83F3E485, 0x1F854511, 0x33FAE87B, 0xA6949A1E,
	0xDB1DA3C5, 0xD5986A2F, 0xAF6CC662, 0xFB04BD0F,
	0x573F3084, 0xDFB9A74F, 0xF60A05B6, 0xDF57E7C7,
	0x11723659, 0x8EAE9859, 0xE08EB664, 0xD310802A,
	0x498B2FA5, 0x555452F1, 0xE5575C26,
]

# This one is set to a potentially non-zero value at the start.
mask4 = [
	0xB6F2BA0C, 0x1A88BEF7, 0x0DDC276B, 0xB000F810,
	0x6D254A35, 0x4C69572C, 0x0E49CFC2, 0xA4391D94,
	0x98E199E1, 0x10F88349, 0xA7CCE380, 0xF4287C11,
	0x93857987, 0xF1F1171F, 0x3487B1FA, 0x65B88727,
	0xA1711A9C, 0x2338B1FE, 0xF742B4DF, 0xA33235EC,
	0x5350D303, 0xC9C1BE55, 0xC4FD2E1D, 0x8039E1DB,
	0xB6386F71, 0x34CBB389, 0x1224001A, 0x59CDEA11,
	0x5371F238,
]

def printd(data: bytearray):
	for b in data:
		print(b, end=', ')
	print('')

def decrypt_in_place(data: bytearray, magic: int) -> bool:
	section_magic, = struct.unpack_from("<I", data)

	m4 = 0
	while m4 < len(mask4):
		decrypted_magic = section_magic ^ ((mask4[m4] + mask1[0] + mask2[0] + mask3[0]) & 0xFFFFFFFF)
		if magic == decrypted_magic:
			break
		m4 += 1

	if m4 == len(mask4):
		return False

	# Encryption and decryption are the same thing
	return encrypt_in_place(data, m4)

def encrypt_in_place(data: bytearray, m4: int) -> bool:
	if m4 < 0 or m4 >= len(mask4):
		raise Exception("Invalid mask ID for encryption.")

	m1 = m2 = m3 = 0

	for i in range(0, len(data), 4):
		v, = struct.unpack_from("<I", data, i)

		key = (mask1[m1] + mask2[m2] + mask3[m3] + mask4[m4]) & 0xFFFFFFFF

		struct.pack_into("<I", data, i, v ^ key)

		m1 += 1
		if m1 >= len(mask1):
			m1 = 0
			m2 += 1
			if m2 >= len(mask2):
				m2 = 0
				m3 += 1
				if m3 >= len(mask3):
					m3 = 0
					m4 = (m4 + 1) % len(mask4)

	return True

def checksum_zero(data: bytearray) -> int:
	b1 = data[-4]
	b2 = data[-3]
	data[-4] = 0
	data[-3] = 0

	c = 0
	for byte in data:
		for b in range(8):
			f = (c & 0x8000) != 0
			c = (c << 1) & 0xFFFF
			if f:
				c ^= 0x1021
			c ^= (byte >> (7 - b)) & 1

	data[-4] = b1
	data[-3] = b2

	return c

def validate_checksum(data: bytearray, magic: int = 0) -> bool:
	return checksum_zero(data) == 0

# --- Ghost data handling ----------------------------------------------
GHOST_SECTION_MAGIC = struct.unpack("<I", b"NKPG")[0]

GHOSTS_ADDR = 0xAF00
GHOST_SIZE = 0xE00

def get_normal_ghost_data(data: bytearray, ordered_course_id: int) -> bytearray:
	if ordered_course_id < 0 or ordered_course_id > 31:
		raise Exception("Invalid course ID given. Expected 0-31.")

	addr = GHOSTS_ADDR + GHOST_SIZE * ordered_course_id
	ghost_data = data[addr:addr + GHOST_SIZE]

	if not decrypt_in_place(ghost_data, GHOST_SECTION_MAGIC):
		raise Exception("Ghost data from save file is invalid.")

	if not validate_checksum(ghost_data, GHOST_SECTION_MAGIC):
		raise Exception("Ghost data from save file is invalid.")

	return ghost_data

def data_to_json(data: bytearray, file: str):
	character_id = data[0x04] & 0xf
	kart_id = (struct.unpack_from('<H', data, 4)[0] >> 4) & 0x3f
	course_id = (struct.unpack_from('<H', data, 4)[0] >> 10) & 0x3f
	inputs = data[0x30:GHOST_SIZE-4]

	json_data = {
		'characterId': character_id,
		'kartId': kart_id,
		'courseId': course_id,
		'ghostInputs': [x for x in inputs],
	}
	content = json.dumps(json_data)
	with open(file, 'w') as fs:
		fs.write(content)

COURSE_ABBREVIATIONS = [
	'F8C', 'YF', 'CCB', 'LM',
	'DH', 'DS', 'WP', 'SR',
	'DKP', 'TTC', 'MC', 'AF',
	'WS', 'PG', 'BC', 'RR',
	'rMC1', 'rMMF', 'rPC', 'rLC1',
	'rDP1', 'rFS', 'rBC2', 'rBP',
	'rKB2', 'rCM', 'rLC2', 'rMB',
	'rCI2', 'rBB', 'rSG', 'rYC'
]

if __name__ == '__main__':
	save_path = sys.argv[1]
	course = sys.argv[2]
	out_path = sys.argv[3]

	contents: bytearray
	with open(save_path, 'rb') as fs:
		contents = bytearray(fs.read())

	if course == 'all':
		for i in range(32):
			ghost_data = get_normal_ghost_data(contents, i)
			data_to_json(ghost_data, f'{out_path}{COURSE_ABBREVIATIONS}.json')
	else:
		course = int(course)
		ghost_data = get_normal_ghost_data(contents, course)
		if not out_path.endswith('.json'):
			out_path += '.json'
		data_to_json(ghost_data, out_path)
