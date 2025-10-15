from __future__ import annotations
import cairo
import numpy as np
import cv2
import threading
from pynput import keyboard
from desmume.emulator import DeSmuME, SCREEN_WIDTH, SCREEN_HEIGHT
from desmume.controls import Keys, keymask

def main():
    # ----------------------------
    # Initialize emulator
    # ----------------------------
    emu = DeSmuME()
    emu.open("mariokart_ds.nds")
    emu.movie.play("baby_park.dsm")
    emu.volume_set(0)
    
    # ----------------------------
    # Keyboard mapping
    # ----------------------------
    KEY_MAP = {
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
    
    pressed_keys = set()
    lock = threading.Lock()
    
    
    # ----------------------------
    # Keyboard event handlers
    # ----------------------------
    def on_press(key):
        try:
            k = key.char if hasattr(key, "char") else key.name
            if k in KEY_MAP:
                with lock:
                    pressed_keys.add(k)
        except Exception:
            pass
    
    
    def on_release(key):
        try:
            k = key.char if hasattr(key, "char") else key.name
            if k in KEY_MAP:
                with lock:
                    pressed_keys.discard(k)
            if key == keyboard.Key.esc:
                return False  # Stop listener
        except Exception:
            pass
    
    
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    
    # ----------------------------
    # Main loop
    # ----------------------------
    running = True
    while running and listener.is_alive():
        emu.cycle()
    
        # Get emulator framebuffer (both screens stacked vertically)
        fb = emu.display_buffer_as_rgbx()
        frame = np.frombuffer(fb, np.uint8).reshape((SCREEN_HEIGHT * 2, SCREEN_WIDTH, 4))
    
        # Extract only top screen
        top_screen = frame[:SCREEN_HEIGHT].copy()
    
        # Cairo surface over just the top screen
        surface = cairo.ImageSurface.create_for_data(
            top_screen, cairo.FORMAT_RGB24, SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_WIDTH * 4
        )
        ctx = cairo.Context(surface)
    
        
    
        # Display top screen only
        cv2.imshow("MKDS Top Screen", top_screen)
        if cv2.waitKey(1) & 0xFF == 27:
            running = False
    
        # Update emulator keymask
        with lock:
            mask = 0
            for k in pressed_keys:
                mask |= keymask(KEY_MAP[k])
        emu.input.keypad_update(mask)
    
    listener.stop()
    cv2.destroyAllWindows()
    
    
if __name__ == __main__:
    main()