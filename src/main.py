from __future__ import annotations
import os, torch, gi, pynput
gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gtk, Gdk, GLib
from src.core.emulator import MarioKart, FX32_SCALE_FACTOR
from src.core.window import (
    EmulatorWindow, 
    SensorOverlay, 
    CollisionOverlay, 
    CheckpointOverlay, 
    OrientationOverlay, 
    DriftOverlay
)
from desmume.controls import Keys, keymask
from contextlib import nullcontext
from argparse import ArgumentParser
from datetime import datetime
import ctypes

from private.mkdslib import *

DEFAULT_USER_FPS = 60.0
DEFAULT_HEADLESS_FPS = 240.0

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


def run_emulator(overlays, args):
    listener, input_state = start_keyboard_listener()
    # Initialize the emulator
    has_grad = args.record or args.record_data
    device = torch.device("cpu")
    emu = MarioKart(
        n_rays=32,
        max_dist=3000.0,
        has_grad=has_grad,
        device=device
    )
    emu.open(args.rom_path)
    emu.volume_set(0) # mute volume so I don't go INSANE

    # Emulator speed
    fps = 60.0 # default fps when not playing .dsm
    if args.fps is not None:
        fps = float(args.fps)

    # DeSmuME movie replay
    #listener, input_state = None, None # realtime keyboard input unnecessary during movie playback
    if args.movie is not None:
        emu.movie.play(args.movie)

        # DeSmuME ROM Save file (critical for loading external .dsm files)
        if args.sram is not None:
            emu.backup.import_file(args.sram)
        
    # DeSmuME movie record
    if args.record is not None:
        emu.movie.record(args.record, author_name="test", sram_save=args.sram)

    # DeSmuME load from savestate
    if args.savestate is not None:
        assert args.movie is None, "savestates cannot be loaded during movie replays"
        emu.savestate.load_file(args.savestate)

    # Record training data
    if args.record_data is not None:
        # speed up fps by default if recording training data from desmume movie replay
        fps: float = fps if args.fps is not None and args.movie is None else 240.0
        overlays = overlays if args.force_overlay else []
        # specify the directory that stores all training directories
        output_dir = args.record_data if args.record_data != '.' else os.getcwd()
    else:
        output_dir = os.getcwd()

    assert fps > 0.0, "fps must be greater than 0"

    # Extra logic for recording training data, specifically for saving all .dat files
    if args.movie is not None:
        output_base = os.path.basename(args.movie).split('.')[0]
    else:
        output_base = f"TRAINING_{datetime.today().strftime('%Y%m%d')}"
    
    dataset_path = f"{output_dir}/{output_base}"
    if not os.path.exists(dataset_path) and args.record_data is not None:
        os.makedirs(dataset_path)

    samples_path = f"{dataset_path}/samples.dat"
    targets_path = f"{dataset_path}/targets.dat"
    metadata_path = f"{dataset_path}/metadata.json"

    window = EmulatorWindow(emu, overlays, device)

    print(
f"""
DeSmuME Savestate:           {args.savestate if args.savestate else "None"}
DeSmuME Movie (loading):     {args.movie if args.movie else "None"}
DeSmuME Movie (saving):      {args.record if args.record else "None"}
Training Data Directory:     {dataset_path if args.record_data else "None"}
FPS:                         {fps:.1f}
    """)

    # Expect the race to end when recording training data
    kill_on_race_complete = args.record_data or args.record
    end_reached = not kill_on_race_complete

    prompt_new_savestate = False
    tmp_save_state_loaded = False
    def tick():
        nonlocal input_state, end_reached, prompt_new_savestate, tmp_save_state_loaded
        if not window.running and emu.memory.race_ready:
            window.show_overlays()
            GLib.idle_add(window.toggle_display_mode)

        GLib.idle_add(emu.cycle)
        window.game_area.queue_draw()

        if emu.memory.race_ready and kill_on_race_complete:
            progress = emu.memory.race_status.driverStatus[0].raceProgress
            progress *= FX32_SCALE_FACTOR
            if progress > 1.0 or emu.movie.is_finished():
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
                elif key == 't':
                    GLib.idle_add(window.toggle_display_mode)
                elif key == 'm':
                    prompt_new_savestate = True
                    GLib.idle_add(window.on_window_destroy)
                elif key == "r":
                    emu.savestate.load(0)
        except:
            pass

        if not emu.memory.race_ready:
            return True
            
        # memory debugging code comes below here

        # to here
        return True

    refresh_rate = int(1000.0 // fps) # Hz
    with (open(samples_path, "wb") if args.record_data is not None else nullcontext() as sf,
          open(targets_path, "wb") if args.record_data is not None else nullcontext() as tf,
          open(metadata_path, "w") if args.record_data is not None else nullcontext() as mf):
        if all([sf, tf, mf]):
            # enable recording of training data
            emu.enable_file_io(sf, tf, mf)
        
        GLib.timeout_add(refresh_rate, tick)
        Gtk.main()

        listener.stop() # end keyboard listener
        
        if kill_on_race_complete and not end_reached:
            print("Warning: User did not complete race. Recorded data/inputs may be incomplete.")
        
        if prompt_new_savestate:
            savestate_filename = input("Savestate file name (must end in .dst): ")
            emu.savestate.save_file(savestate_filename)

        if args.sram is not None:
            emu.backup.export_file(args.sram)

        emu.close() # safely stop emulator

def add_recording_offset(dsm_path, frame_offset, out_path=None):
    if out_path is None:
        base, ext = dsm_path.split('.')[0], dsm_path.split('.')[1]
        out_path = f"{base}_offset_{frame_offset}.{ext}"

    

def main(overlays=[]):
    parser = ArgumentParser(
        prog="main.py",
        description="Runs a user friendly visualization of mariokart ds",
    )
    parser.add_argument('-p', '--rom-path', default="private/mariokart_ds.nds")
    parser.add_argument('-r', '--record')
    parser.add_argument('-d', '--record-data')
    parser.add_argument('-m', '--movie')
    parser.add_argument('-s', '--savestate')
    parser.add_argument('--correct-movie', action='store_true')
    parser.add_argument('--sram')
    parser.add_argument('--fps')
    parser.add_argument('--force-overlay', action='store_true')
    args = parser.parse_args()
    
    run_emulator(overlays, args)
    
    


OVERLAYS = [
    CollisionOverlay,
    CheckpointOverlay,
    SensorOverlay,
    DriftOverlay,
    OrientationOverlay
]


if __name__ == "__main__":
    main(overlays=OVERLAYS)
