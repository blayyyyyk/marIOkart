from desmume.controls import keymask, Keys
from pynput import keyboard

emu = None

key_map = {
    "a": Keys.KEY_LEFT,
    "d": Keys.KEY_RIGHT,
}


def on_press(key):
    global emu, key_map
    assert emu is not None
    try:
        name = key.char.lower()
    except AttributeError:
        name = key.name
    if name in key_map:
        _keymask = keymask(key_map[name])
        emu.input.keypad_add_key(_keymask)


def on_release(key):
    global emu, key_map
    assert emu is not None
    try:
        name = key.char.lower()
    except AttributeError:
        name = key.name
    if name in key_map:
        _keymask = keymask(key_map[name])
        emu.input.keypad_rm_key(_keymask)


def init_keyhack(emulator):
    global emu
    emu = emulator
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
