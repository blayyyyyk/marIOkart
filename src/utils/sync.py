from __future__ import annotations
import os, torch, gi, pynput, re, os
gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gtk, Gdk, GLib
from src.core.emulator import MarioKart
from src.display.window import EmulatorWindow
from argparse import ArgumentParser
from mkdslib.mkdslib import *
from src.main import DEFAULT_USER_FPS, DEFAULT_HEADLESS_FPS

BOOST_WINDOW_START = 193
BOOST_WINDOW_END = 218


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

def coarse_correction(emu: MarioKart, window: EmulatorWindow, args, device):
    emu.movie.play(args.movie)

    if args.sram is not None:
        emu.backup.import_file(args.sram)

    sync_stats = {
        "offset": 0,
        "index": 0,
        "synced": False
    }

    race_start_time = -1
    def tick():
        nonlocal sync_stats, race_start_time
        if not window.running and emu.memory.race_ready:
            window.show_overlays()
            GLib.idle_add(window.toggle_display_mode)

        GLib.idle_add(emu.cycle)
        
        window.game_area.queue_draw()
        if not emu.memory.race_ready:
            return True
        elif not window.running:
            return False
        
        if race_start_time < 0:
            race_start_time = emu.count
        
        progress = emu.memory.driver_status.raceProgress
        if progress >= 1.0 or emu.movie.is_finished():
            GLib.idle_add(window.on_window_destroy)
            
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
            GLib.idle_add(window.on_window_destroy)

        return True

    fps = float(args.fps)
    refresh_rate = int(1000.0 // fps) # Hz
    GLib.timeout_add(refresh_rate, tick)
    Gtk.main()

    return sync_stats


def fine_correction(emu: MarioKart, window: EmulatorWindow, args, device):
    emu.movie.play(args.movie)
    
    sync_stats = {
        "progress": 0.0
    }
    def tick():
        nonlocal sync_stats
        if not window.running and emu.memory.race_ready:
            window.show_overlays()
            GLib.idle_add(window.toggle_display_mode)
    
        GLib.idle_add(emu.cycle)
        
        window.game_area.queue_draw()
        if not emu.memory.race_ready:
            return True
        
        driver = emu.memory.driver.position
        progress = emu.memory.driver_status.raceProgress
        if progress >= 1.0 or emu.movie.is_finished():
            sync_stats['progress'] = progress
            return False
    
        return True
    
    fps = float(args.fps)
    refresh_rate = int(1000.0 // fps) # Hz
    GLib.timeout_add(refresh_rate, tick)
    Gtk.main()
    GLib.idle_add(window.toggle_display_mode)
    
    return sync_stats


def frame_correction_sync(args):
    # Initialize the emulator
    device = torch.device("cpu")
    emu = MarioKart(device=device)
    emu.open(args.rom_name)
    emu.volume_set(0)
    window = EmulatorWindow(emu, [], device)
    
    # Initial run to align movie to beginning of speed boost window
    result_stats = None
    editor = MovieEditor(args.movie)
    if not args.check:
        result_stats = coarse_correction(emu, window, args, device)
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
        progress = fine_correction(emu, window, args, device)['progress']
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
    
    GLib.idle_add(window.on_window_destroy)
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