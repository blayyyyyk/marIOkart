import math, os, sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.emulator import init_desmume
from utils.keyboard import init_keyhack
from utils.memory import read_vector_2d, ADDR_ORIENTATION_X, ADDR_ORIENTATION_Z


emu, window = init_desmume("../mariokart_ds.nds")
init_keyhack(emu)


def main():
    # Run the emulation as fast as possible until quit
    while not window.has_quit():
        window.process_input()  # Controls are the default DeSmuME controls, see below.
        emu.cycle()
        window.draw()

        dir_x, dir_z = read_vector_2d(
            emu.memory.unsigned, ADDR_ORIENTATION_X, ADDR_ORIENTATION_Z
        )
        angle_rad = math.atan2(dir_z, dir_x) * 2
        angle_deg = math.degrees(angle_rad)

        os.system("clear")
        print(f"Cart Orientation: {angle_rad} radians / {angle_deg} degrees")


if __name__ == "__main__":
    main()
