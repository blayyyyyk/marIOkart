from __future__ import annotations
from dataclasses import dataclass
from multiprocessing import Process, freeze_support
from uuid import UUID
from desmume.frontend.gtk_drawing_impl.software import SCREEN_HEIGHT, SCREEN_WIDTH
import os, torch, gi, pynput

from src.utils.recording import extract_ghost

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gtk, Gdk, GLib
from src.core.emulator import MarioKart, FX32_SCALE_FACTOR
# from src.display.window import (
#     EmulatorWindow,
#     RealWindow,
#     VirtualWindow,
# )
from src.display.overlay import (
    SensorOverlay,
    CollisionOverlay,
    CheckpointOverlay,
    OrientationOverlay,
    DriftOverlay,
)
#from src.display.multiwindow import MultiEmulatorWindow
from src.display.window import EmulatorBackend, EmulatorCallable, EmulatorConfig, EmulatorFrontend, WindowBackend
from desmume.controls import Keys, keymask
from contextlib import nullcontext
from argparse import ArgumentParser
from datetime import datetime
import ctypes
from os import listdir
from os.path import isfile, join
from pathlib import Path
from typing import Any, Literal, TypedDict, Unpack, Optional, cast
from src.utils.recording.extract_ghost import (
    get_normal_ghost_data,
    has_ghost,
    data_to_dict,
    COURSE_ABBREVIATIONS,
)
from src.utils.recording.convert_to_dsm import convert_to_dsm
from src.mkdslib.mkdslib import *
from multiprocessing.shared_memory import SharedMemory

DEFAULT_USER_FPS = 60.0
DEFAULT_HEADLESS_FPS = 240.0
SCREEN_SCALE = 3
SCREEN_ORIENTATION = "horizontal"

OVERLAYS = [
    CollisionOverlay,
    CheckpointOverlay,
    SensorOverlay,
    DriftOverlay,
    OrientationOverlay,
]

# key mapping
USER_KEYMAP = {
    "w": Keys.KEY_UP,
    "s": Keys.KEY_DOWN,
    "a": Keys.KEY_LEFT,
    "d": Keys.KEY_RIGHT,
    "z": Keys.KEY_B,
    "x": Keys.KEY_A,
    "u": Keys.KEY_X,
    "i": Keys.KEY_Y,
    "q": Keys.KEY_L,
    "e": Keys.KEY_R,
    " ": Keys.KEY_START,
    "left": Keys.KEY_LEFT,
    "right": Keys.KEY_RIGHT,
    "up": Keys.KEY_UP,
    "down": Keys.KEY_DOWN,
}




# example entry config type signature
@dataclass
class MainKwargs(EmulatorConfig):
    scale: Optional[int]
    native_scale: Optional[int]

def main(
    emu: MarioKart,
    config: MainKwargs,
    input_state: Optional[set[str]]
):
    if emu.memory.race_ready:
        progress = emu.memory.race_status.driverStatus[0].raceProgress
        progress *= FX32_SCALE_FACTOR
        if progress > 1.0 or emu.movie.is_finished():
            if emu.file_io is not None:
                emu.file_io.close(None)

            end_reached = True
            return False

    
    emu.cycle()
    emu.input.keypad_update(0)

    try:
        if input_state:
            for key in input_state:
                if key in USER_KEYMAP:
                    emu.input.keypad_add_key(keymask(USER_KEYMAP[key]))
                elif key == "r":
                    emu.savestate.load(0)
    except:
        pass

    if not emu.memory.race_ready:
        return True

    # memory debugging code comes below here

    # to here
    return True
    
    
        
SCALE = 2
NATIVE_SCALE = 1
def batch(func: EmulatorCallable[MainKwargs], batch_config: list[MainKwargs]):
    # make a new emulation process for each config
    backend = EmulatorBackend(batch_config[0].native_scale or NATIVE_SCALE)
    
    for config in batch_config:
        backend.add_instance(func, config, OVERLAYS)
    
    backend.start()
    # attach all shared display buffers to a gtk window
    frontend = EmulatorFrontend(backend, batch_config[0].scale or SCALE)
    frontend.start()
    
    

if __name__ == "__main__":
    parser = ArgumentParser(
        prog="main.py",
        description="Main entry point for the Mario Kart DS ML visualization application.",
    )

    # Main parser #
    parser.add_argument(
        "-p",
        "--rom-path",
        default="private/mariokart_ds.nds",
        help="Path to the Mario Kart DS ROM file.",
        type=Path,
    )
    parser.add_argument("-r", "--record", help="Path to record gameplay to.", type=Path)
    parser.add_argument(
        "-d", "--record-data", help="Path to record gameplay data to.", type=Path
    )
    parser.add_argument(
        "-m", "--movie", help="Path to a movie file to playback.", type=Path
    )
    parser.add_argument(
        "-s", "--savestate", help="Path to a savestate file to load.", type=Path
    )
    parser.add_argument("--sram", help="Path to SRAM file.")
    parser.add_argument("--fps", help="Target frames per second.")
    parser.add_argument(
        "--force-overlay", action="store_true", help="Flag to force overlay display."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Flag to enable verbose output."
    )
    parser.add_argument("--model", help="Path to a model file to load.", type=Path)

    parser.set_defaults(func=main)
    args = parser.parse_args()
    
    if args.movie:
        if os.path.isdir(args.movie):
            movie_dir = args.movie
            movie_files = [
                f for f in listdir(str(movie_dir)) if isfile(join(str(movie_dir), f))
            ]
            configs = []
            for movie_f in movie_files:
                kwargs = args.__dict__.copy()
                del kwargs['func']
                config = MainKwargs(**kwargs, native_scale=1, scale=1)
                config.movie = Path(f"{movie_dir}/{movie_f}")
                config.verbose = False
                configs.append(config)
    
            batch(args.func, configs)
            exit()
    
    kwargs = args.__dict__.copy()
    del kwargs['func']
    config = MainKwargs(**kwargs, native_scale=3, scale=3)
    batch(args.func, [config])
