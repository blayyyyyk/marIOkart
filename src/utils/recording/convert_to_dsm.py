from desmume.emulator import StartFrom
from desmume.controls import Keys, keymask

from src.core.memory import *
from src.utils.desmume_ext import DeSmuME

import json
import os
import sys

# Array of tuples. Each tuple is (MKDS button, DSM letter). In the order that they appear in movie files.
# 0x100 as mkds button indicates a button not used by ghosts
MKDS_TO_DESMUME_BUTTONS = [
	(0x10, 'R'),
	(0x20, 'L'),
	(0x80, 'D'),
	(0x40, 'U'),
	(0x100, 'T'),
	(0x100, 'S'),
	(0x02, 'B'),
	(0x01, 'A'),
	(0x100, 'Y'),
	(0x08, 'X'),
	(0x100, 'W'),
	(0x04, 'E'),
	(0x100, 'g'),
]

SAVE_FILE_LOCATION = 'src/utils/recording/100percent.sav'

def get_scene_state(emu: DeSmuME):
	addr = emu.memory.unsigned.read_long(SCENE_STATE_PTR_ADDR)
	if addr == 0:
		return 0
	else:
		addr += 0x10C
		return emu.memory.unsigned[addr]
def get_race_timer(emu: DeSmuME):
	addr = emu.memory.unsigned.read_long(RACE_STATE_PTR_ADDR)
	return emu.memory.signed.read_long(addr + 4) # frameCounter

def validate(emu: DeSmuME, data: dict):
	raceConfig = emu.memory.unsigned.read_long(RACE_CONFIG_PTR_ADDR)
	playerConfig = raceConfig + 0x68
	if data['characterId'] != emu.memory.unsigned[playerConfig]:
		return False
	if data['kartId'] != emu.memory.unsigned[playerConfig + 4]:
		return False
	if data['courseId'] != emu.memory.unsigned[raceConfig]:
		return False

	return True

def do_menu(emu: DeSmuME, character: int, kart: int, course: int):
	def press(key):
		emu.input.keypad_update(keymask(key))
		emu.cycle(False)
		emu.input.keypad_update(0)
	def wait(frames):
		for i in range(frames):
			emu.cycle(False)

	print("Menuing...")
	# Title screen and main menu
	wait(360)
	press(Keys.KEY_A)

	# Game mode selection
	wait(110)
	press(Keys.KEY_DOWN)
	press(Keys.KEY_A)
	wait(45)
	press(Keys.KEY_A)

	# Character select
	wait(45)

	CHARACTER_ORDER = [ 0, 7, 4, 6, 2, 1, 5, 3, 9, 8, 10, 11 ]
	charcter_by_order = CHARACTER_ORDER.index(character)
	character_x = charcter_by_order % 4
	character_y = charcter_by_order // 4
	for i in range(character_x):
		press(Keys.KEY_RIGHT)
		wait(7)
	for i in range(character_y):
		press(Keys.KEY_DOWN)
		wait(7)
	press(Keys.KEY_A)
	wait(30)

	# Kart select
	KART_ORDER = [ # Is the character's 1-shroomer their default kart?
		False, False, True, False,
		False, True, True, True,
		False, False, True, True,
	]
	owning_character_id = kart // 3
	base_kart_order_placement = CHARACTER_ORDER.index(owning_character_id) * 3
	base_kart_order_placement -= CHARACTER_ORDER.index(character) * 3
	if base_kart_order_placement < 0:
		base_kart_order_placement += 36
	remaining_rights = 0
	if kart % 3 == 0:
		remaining_rights = 1
	elif (kart % 3 == 2) == KART_ORDER[owning_character_id]:
		remaining_rights = 2
	for i in range(base_kart_order_placement + remaining_rights):
		press(Keys.KEY_RIGHT)
		wait(7)
	press(Keys.KEY_A)

	# Course select
	wait(160)

	COURSE_ID_TO_ORDERED = [
		-1, 31, 37, 38, 39, -1, 6, 12,
		40, 23, 16, 17, 22, 18, 19, 24,
		21, 9, 3, 11, 0, 41, 1, 29,
		7, 10, 13, 4, 5, 15, 8, 2,
		14, 6, 12, 20, 25, 26, 27, 28,
		30, 32, 32, 32, 32, 32, 32, 32,
		44, 45, 46, 47, 48, 49, 50, -1,
	]
	ordered_course_number = COURSE_ID_TO_ORDERED[course]
	assert ordered_course_number >= 0 and ordered_course_number < 32, "Invalid course ID"
	cup = ordered_course_number // 4
	cup_x = cup % 4
	cup_y = cup // 4
	for i in range(cup_x):
		press(Keys.KEY_RIGHT)
		wait(1)
	for i in range(cup_y):
		press(Keys.KEY_DOWN)
		wait(1)
	press(Keys.KEY_A)
	wait(55)
	for i in range(ordered_course_number % 4):
		press(Keys.KEY_DOWN)
		wait(1)
	press(Keys.KEY_A)
	wait(15)
	press(Keys.KEY_A)

	# Wait for the race to start
	MAX_FRAMES_TO_WAIT = 300
	frames = 0
	while emu.memory.unsigned.read_long(RACER_PTR_ADDR) == 0 and frames < MAX_FRAMES_TO_WAIT:
		emu.cycle(False)
		frames += 1
	while get_scene_state(emu) != 2:
		emu.cycle(False)
		frames += 1
	if frames >= MAX_FRAMES_TO_WAIT:
		emu.movie.stop()
		raise Exception('Race did not start! Something went wrong with the menuing.')

def run(input_file:str, output_file: str | None = None):
	output_file = output_file or input_file[:-4] + 'dsm'
	content: str
	with open(input_file, 'r') as fs:
		content = fs.read()
	data = json.loads(content)

	inputs = data['ghostInputs']
	assert len(inputs) == 0xdcc

	emu = DeSmuME()
	emu.open('private/mariokart_ds.nds')
	emu.backup.import_file(SAVE_FILE_LOCATION)
	emu.movie.record(output_file, 'marIOkart', StartFrom.START_SRAM, SAVE_FILE_LOCATION)
	emu.input.keypad_update(0)
	emu.volume_set(0)

	do_menu(emu, data['characterId'], data['kartId'], data['courseId'])

	# Just make sure we're on the first frame
	while get_race_timer(emu) != 1:
		emu.cycle(False)

	start_frame = emu.movie.get_length()
	emu.movie.stop()
	if not validate(emu, data):
		raise Exception('We somehow got the wrong combo or course. Something went wrong with the menuing.')

	print('Adding inputs and metadata...')

	blank_frame = '|0|.............000 000 0 000|\n'
	movie_contents: list[str]
	with open(output_file, 'r') as fs:
		movie_contents = fs.readlines()
	# Wait for the first ghost input
	for i in range(120):
		movie_contents.append(blank_frame)

	# Write ghost inputs
	input_total_count = (inputs[0] | (inputs[1] << 8)) // 2
	assert input_total_count <= 1764
	input_count = 0
	index = input_count*2 + 4
	while input_count < input_total_count:
		buttons = inputs[index]
		frame_count = inputs[index + 1]
		frame_inputs = '|0|'
		for tuple in MKDS_TO_DESMUME_BUTTONS:
			if buttons & tuple[0] != 0:
				frame_inputs += tuple[1]
			else:
				frame_inputs += '.'
		frame_inputs += '000 000 0 000|\n'

		for i in range(frame_count):
			movie_contents.append(frame_inputs)

		input_count += 1
		index = input_count*2 + 4

	# Put our data in the comment header
	index = 0
	while not movie_contents[index].startswith('comment '):
		index += 1
		if index > 20:
			raise Exception('Movie has no comment?')
	movie_contents[index] = f"comment {data['characterId']} {data['kartId']} {data['courseId']} {start_frame}\n"

	with open(output_file, 'w') as fs:
		fs.writelines(movie_contents)


if __name__ == "__main__":
	run(sys.argv[1])
