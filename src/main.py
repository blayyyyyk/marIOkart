from __future__ import annotations
from desmume.frontend.gtk_drawing_impl.software import SCREEN_HEIGHT, SCREEN_WIDTH
import os, torch, gi, pynput

from src.utils.recording import extract_ghost

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gtk, Gdk, GLib
from src.core.emulator import MarioKart, FX32_SCALE_FACTOR
from src.core.window import (
    EmulatorWindow,
    RealWindow,
    VirtualWindow,
    SensorOverlay,
    CollisionOverlay,
    CheckpointOverlay,
    OrientationOverlay,
    DriftOverlay,
)
from desmume.controls import Keys, keymask
from contextlib import nullcontext
from argparse import ArgumentParser
from datetime import datetime
import ctypes
from os import listdir
from os.path import isfile, join
from pathlib import Path
from typing import Literal, TypedDict, Unpack, Optional
from src.utils.recording.extract_ghost import (
    get_normal_ghost_data,
    has_ghost,
    data_to_dict,
    COURSE_ABBREVIATIONS,
)
from src.utils.recording.convert_to_dsm import convert_to_dsm
from src.mkdslib.mkdslib import *

DEFAULT_USER_FPS = 60.0
DEFAULT_HEADLESS_FPS = 240.0
SCREEN_SCALE = 1
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


# async keyboard handler
def start_keyboard_listener():
    """Starts a non-blocking keyboard listener in a separate thread."""
    input_state = set()

    def on_press(key):
        try:
            name = key.char.lower() if hasattr(key, "char") else key.name
            input_state.add(name)

        except Exception:
            pass

    def on_release(key):
        try:
            name = key.char.lower() if hasattr(key, "char") else key.name
            input_state.discard(name)
        except Exception:
            pass

    listener = pynput.keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.daemon = True
    listener.start()

    return listener, input_state


class MainKwargs(TypedDict):
    rom_path: Path
    record: Optional[Path]
    record_data: Optional[Path]
    fps: Optional[float]
    movie: Optional[Path]
    sram: Optional[Path]
    savestate: Optional[Path]
    force_overlay: bool
    id: Optional[int]


def main(
    emu: Optional[MarioKart] = None,
    window: Optional[VirtualWindow | RealWindow] = None,
    **kwargs: Unpack[MainKwargs],
):
    overlays = OVERLAYS
    print(overlays)
    listener, input_state = start_keyboard_listener()
    # Initialize the emulator
    has_grad = kwargs["record"] is not None or kwargs["record_data"] is not None
    DEVICE = torch.device("cpu")
    if emu is None:
        emu = MarioKart(n_rays=32, max_dist=3000.0, has_grad=has_grad, device=DEVICE)
        emu.open(str(kwargs["rom_path"]))
        emu.volume_set(0)  # mute volume so I don't go INSANE
    else:
        emu.reset()

    if window is None:
        window = EmulatorWindow(
            emu,
            overlays,
            width=SCREEN_WIDTH,
            height=SCREEN_HEIGHT,
            orientation=SCREEN_ORIENTATION,
            scale=SCREEN_SCALE,
            device=DEVICE,
            virtual_screen=True,
            id=kwargs["id"] if kwargs["id"] is not None else 0,
        )
        window.show_all()

    # Emulator speed
    fps = 60.0  # default fps when not playing .dsm
    if kwargs["fps"] is not None:
        fps = float(kwargs["fps"])

    # DeSmuME movie replay
    # listener, input_state = None, None # realtime keyboard input unnecessary during movie playback
    if kwargs["movie"] is not None:
        emu.movie.play(str(kwargs["movie"]))

        # DeSmuME ROM Save file (critical for loading external .dsm files)
        if kwargs["sram"] is not None:
            emu.backup.import_file(str(kwargs["sram"]))

    # DeSmuME movie record
    if kwargs["record"] is not None:
        emu.movie.record(
            str(kwargs["record"]), author_name="test", sram_save=str(kwargs["sram"])
        )

    # DeSmuME load from savestate
    if kwargs["savestate"] is not None:
        assert (
            kwargs["movie"] is None
        ), "savestates cannot be loaded during movie replays"
        emu.savestate.load_file(str(kwargs["savestate"]))

    # Record training data
    if kwargs["record_data"] is not None:
        # speed up fps by default if recording training data from desmume movie replay
        fps: float = (
            fps if kwargs["fps"] is not None and kwargs["movie"] is None else 240.0
        )
        overlays = overlays if kwargs["force_overlay"] else []
        # specify the directory that stores all training directories
        output_dir = (
            str(kwargs["record_data"])
            if str(kwargs["record_data"]) != "."
            else os.getcwd()
        )
    else:
        output_dir = os.getcwd()

    assert fps > 0.0, "fps must be greater than 0"

    # Extra logic for recording training data, specifically for saving all .dat files
    if kwargs["movie"] is not None:
        output_base = os.path.basename(kwargs["movie"]).split(".")[0]
    else:
        output_base = f"TRAINING_{datetime.today().strftime('%Y%m%d')}"

    dataset_path = f"{output_dir}/{output_base}"
    if not os.path.exists(dataset_path) and kwargs["record_data"] is not None:
        os.makedirs(dataset_path)

    samples_path = f"{dataset_path}/samples.dat"
    targets_path = f"{dataset_path}/targets.dat"
    metadata_path = f"{dataset_path}/metadata.json"

    print(f"""
DeSmuME Savestate:           {kwargs['savestate'] if kwargs['savestate'] else "None"}
DeSmuME Movie (loading):     {kwargs['movie'] if kwargs['movie'] else "None"}
DeSmuME Movie (saving):      {kwargs['record'] if kwargs['record'] else "None"}
Training Data Directory:     {dataset_path if kwargs['record_data'] else "None"}
FPS:                         {fps:.1f}
    """)

    # Expect the race to end when recording training data
    kill_on_race_complete = kwargs["record_data"] or kwargs["record"] or kwargs["movie"]
    end_reached = not kill_on_race_complete

    prompt_new_savestate = False
    tmp_save_state_loaded = False

    def tick():
        nonlocal input_state, end_reached, prompt_new_savestate, tmp_save_state_loaded
        if not window.running and emu.memory.race_ready:
            window.start_overlay_threads()
            GLib.idle_add(window.toggle_display_mode)

        GLib.idle_add(emu.cycle)
        window.game_area.queue_draw()

        if emu.memory.race_ready and kill_on_race_complete:
            progress = emu.memory.race_status.driverStatus[0].raceProgress
            progress *= FX32_SCALE_FACTOR
            if progress > 1.0 or emu.movie.is_finished():
                if emu._io is not None:
                    emu._io.close(None)

                end_reached = True
                GLib.idle_add(window.on_window_destroy)
        elif emu.memory.race_ready and not tmp_save_state_loaded:
            tmp_save_state_loaded = True
            emu.savestate.save(0)

        emu.input.keypad_update(0)

        try:
            for key in input_state:
                if key in USER_KEYMAP:
                    emu.input.keypad_add_key(keymask(USER_KEYMAP[key]))
                elif key == "t":
                    GLib.idle_add(window.toggle_display_mode)
                elif key == "m":
                    prompt_new_savestate = True
                    GLib.idle_add(window.on_window_destroy)
                elif key == "r":
                    emu.savestate.load(0)
        except:
            pass

        if not emu.memory.race_ready:
            return True

        # memory debugging code comes below here
        # os.system('clear')
        # d = emu.memory.driver
        # print(d.netColPos)

        # to here
        return True

    refresh_rate = int(1000.0 // fps)  # Hz
    with (
        (
            open(samples_path, "wb")
            if kwargs["record_data"] is not None
            else nullcontext()
        ) as sf,
        (
            open(targets_path, "wb")
            if kwargs["record_data"] is not None
            else nullcontext()
        ) as tf,
        (
            open(metadata_path, "w")
            if kwargs["record_data"] is not None
            else nullcontext()
        ) as mf,
    ):
        if all([sf, tf, mf]):
            # enable recording of training data
            emu.enable_file_io(sf, tf, mf)

        GLib.timeout_add(refresh_rate, tick)
        Gtk.main()

        listener.stop()  # end keyboard listener

        if kill_on_race_complete and not end_reached:
            print(
                "Warning: User did not complete race. Recorded data/inputs may be incomplete."
            )

        if prompt_new_savestate:
            savestate_filename = input("Savestate file name (must end in .dst): ")
            emu.savestate.save_file(savestate_filename)

        if kwargs["sram"] is not None:
            emu.backup.export_file(str(kwargs["sram"]))

        return emu


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

    parser.set_defaults(func=main)
    args = parser.parse_args()

    if args.movie is None:
        args.func(**args.__dict__)
        exit()

    if os.path.isdir(args.movie):
        dirname = args.movie
        emu = None
        onlyfiles = [f for f in listdir(str(dirname)) if isfile(join(str(dirname), f))]
        print(f"Movie directory detected, running on {len(onlyfiles)} files")
        for f in onlyfiles:
            args.movie = f"{dirname}/{f}"
            emu = args.func(emu=emu, **args.__dict__)
    elif args.movie.suffix == ".sav":
        print("Save file detected, extracting ghost inputs and converting to dsm.")
        contents: bytearray
        with open(args.movie, "rb") as fs:
            contents = bytearray(fs.read())
            contents = bytearray(fs.read())

        course_count = 0
        for course_id in range(32):
            if has_ghost(contents, course_id):
                continue
            course_count += 1
            ghost_data = get_normal_ghost_data(contents, course_id)
            ghost_data = data_to_dict(ghost_data)
            assert isinstance(args.movie, Path)
            output_file_name = (
                f"{args.movie.stem}_{COURSE_ABBREVIATIONS[course_id]}.dsm"
            )
            convert_to_dsm(output_file_name)
            args.movie = output_file_name
            main(**args.__dict__)
    else:
        args.func(**args.__dict__)
