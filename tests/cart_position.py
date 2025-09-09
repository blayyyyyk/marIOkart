import math, os, sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.emulator import init_desmume
from utils.keyboard import init_keyhack
from utils.memory import read_vector_3d, ADDR_POSITION

emu, window = init_desmume("../mariokart_ds.nds")
init_keyhack(emu)


def main():
    # Run the emulation as fast as possible until quit
    while not window.has_quit():
        window.process_input()  # Controls are the default DeSmuME controls, see below.
        emu.cycle()
        window.draw()

        x, y, z = read_vector_3d(emu.memory.unsigned, ADDR_POSITION)

        os.system("clear")
        print(f"Cart Position: (x) {x} (y) {y} (z) {z}")


if __name__ == "__main__":
    main()
