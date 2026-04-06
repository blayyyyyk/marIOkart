from __future__ import annotations

import os
import re

import gi
import pynput
import torch

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from argparse import ArgumentParser

from gi.repository import Gdk, GLib, Gtk

from ..core.emulator import MarioKart
from ..mkdslib.mkdslib import *

DEFAULT_USER_FPS = 60
DEFAULT_HEADLESS_FPS = 60
BOOST_WINDOW_START = 193
BOOST_WINDOW_END = 220 # 218


pattern_str = r"\|\d{1}\|[0-9A-Z.]{13}\d{3}\s\d{3}\s\d{1}\s\d{3}\|"
pattern = re.compile(pattern_str)

class MovieEditor:
    def __init__(self, file_name, tmp_file_name=None):
        with open(file_name, "r") as f:
            contents = f.readlines()

        self.file_name = file_name
        self.tmp_file_name = f"{self.file_name.split('.')[0]}_tmp.dsm" if tmp_file_name is None else tmp_file_name
        self.header = []
        for d in contents:
            if re.match(pattern, d): break
            self.header.append(d)

        self.contents = contents[len(self.header):]

    def __del__(self):
        if self.tmp_file_name == self.file_name: return
        if os.path.exists(self.tmp_file_name):
            os.remove(self.tmp_file_name)

    def shift_back(self, index, amt):
        left = self.contents[:index-amt-1]
        right = self.contents[index-1:]
        self.contents = left + right

    def shift_forward(self, index, amt):
        padding = "|0|.............000 000 0 000|\n"
        left = self.contents[:index-1]
        mid = [padding] * amt
        right = self.contents[index-1:]
        self.contents = left + mid + right

    def save(self, tmp=True) -> str:
        file_name = self.tmp_file_name if tmp else self.file_name
        with open(file_name, "w") as f:
            f.writelines(self.header + self.contents)

        return file_name

def coarse_correction(emu: MarioKart, args, device):
    emu.movie.play(args.movie)
    window = emu.create_sdl_window()

    if args.sram is not None:
        emu.backup.import_file(args.sram)

    sync_stats = {
        "offset": int(0),
        "index": 0,
        "synced": False
    }

    race_start_time = -1
    while True:
        window.process_input()
        emu.cycle(with_joystick=True)
        window.draw()

        if not emu.memory.race_ready:
            continue

        if race_start_time < 0:
            race_start_time = emu.count

        progress = emu.memory.driver_status.raceProgress
        if progress >= 1.0 or emu.movie.is_finished():
            break

        mask = f"{emu.input.keypad_get():09b}"
        if mask[-1] == '1' and sync_stats['offset'] == 0:
            key_press_time = emu.count
            player_wait_time = key_press_time - race_start_time
            min_diff = BOOST_WINDOW_START - player_wait_time
            max_diff = BOOST_WINDOW_END - player_wait_time

            sync_stats = {
                'offset': min_diff,
                'index': key_press_time,
            }



    return sync_stats


def fine_correction(emu: MarioKart, args, device):
    emu.movie.play(args.movie)
    window = emu.create_sdl_window()


    sync_stats = {
        "progress": 0.0
    }
    while True:
        window.process_input()
        emu.cycle()
        window.draw()

        if not emu.memory.race_ready:
            continue

        progress = emu.memory.driver_status.raceProgress
        if progress >= 1.0 or emu.movie.is_finished():
            sync_stats['progress'] = progress
            break

    return sync_stats


def frame_correction_sync(args):
    print(args.movie)

    # Initialize the emulator
    device = torch.device("cpu")
    emu = MarioKart(device=device)
    emu.open(args.rom_name)
    emu.volume_set(0)

    # Initial run to align movie to beginning of speed boost window
    result_stats = None
    editor = MovieEditor(args.movie)
    if not args.check:
        result_stats = coarse_correction(emu, args, device)
        if result_stats['offset'] < 0:
            editor.shift_back(result_stats['index'], -result_stats['offset'])
        else:
            editor.shift_forward(result_stats['index'], result_stats['offset'])

    original_file_name = args.movie
    tmp_file_name = editor.save(tmp=(not args.check))
    args.movie = tmp_file_name

    # More precise adjustments to ensure race completion
    progress = 0.0
    start_offset = result_stats['index'] + result_stats['offset'] if result_stats is not None else 0
    for frame_offset in range(25):
        print(f"Testing desync at frame {start_offset + frame_offset} (+{frame_offset})...")
        emu.reset()
        progress = fine_correction(emu, args, device)['progress']
        print(f"Testing completed.\nSyncing progress: {progress*100:.2f}%")
        if progress >= 1.0 or result_stats is None: break
        editor.shift_forward(result_stats['index'] + result_stats['offset'], 1)
        editor.save(tmp=True)

    if progress >= 1.0:
        print("Syncing successful!")
        if tmp_file_name != original_file_name:
            if args.out is not None:
                editor.file_name = args.out

            editor.save(tmp=False)
    else:
        print(f"Syncing failed at {progress*100:.2f}%.")

    emu.close()
    return

if __name__ == "__main__":
    parser = ArgumentParser(
        prog="sync.py",
        description="Frame Correction for DeSmuME Movie Desync"
    )

    parser.add_argument('-f', "--rom-name", default="private/mariokart_ds.nds")
    parser.add_argument('-m', "--movie", required=True)
    parser.add_argument('-s', "--sram")
    parser.add_argument('-c', "--check", action="store_true")
    parser.add_argument('-o', "--out", default=None)
    parser.add_argument('--fps', default=None)
    args = parser.parse_args()
    if args.fps is None:
        args.fps =  DEFAULT_USER_FPS if args.check is None else DEFAULT_HEADLESS_FPS

    frame_correction_sync(args)
