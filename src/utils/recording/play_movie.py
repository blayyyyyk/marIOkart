# This script exists just to help diagnose any issues with convert_to_dsm.py

from desmume.emulator import DeSmuME

import sys

emu = DeSmuME()
emu.open('private/mariokart_ds.nds')
emu.movie.play(sys.argv[1])
emu.volume_set(0)

window = emu.create_sdl_window()
while not window.has_quit():
	emu.cycle(False)
	window.draw()
