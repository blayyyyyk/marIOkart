import cairo
import gi
import os
import threading
import queue

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gtk, Gdk, GLib

from desmume.controls import Keys, keymask
from desmume.emulator import DeSmuME
from desmume.frontend.gtk_drawing_area_desmume import AbstractRenderer

# --- Environment setup for Homebrew libraries ---
os.environ['PKG_CONFIG_PATH'] = "/opt/homebrew/lib/pkgconfig:$PKG_CONFIG_PATH"
os.environ['DYLD_FALLBACK_LIBRARY_PATH'] = "/opt/homebrew/lib:$DYLD_FALLBACK_LIBRARY_PATH"

# --- Global variables ---
emu = None
renderer = None
dot_radius = 5
SCREEN_WIDTH, SCREEN_HEIGHT = 256, 192
current_input = 0
points: dict[str, list[list[float]]] = {}

# Queue for callback processing in a separate thread
callback_queue = queue.Queue()


# --- Key handling ---
def on_key_press(widget, event):
    global emu, current_input
    assert emu is not None
    key = Gdk.keyval_name(event.keyval).lower()

    if key in ("w", "up"):
        emu.input.keypad_add_key(keymask(Keys.KEY_UP))
    elif key in ("s", "down"):
        emu.input.keypad_add_key(keymask(Keys.KEY_DOWN))
    elif key in ("a", "left"):
        emu.input.keypad_add_key(keymask(Keys.KEY_LEFT))
    elif key in ("d", "right"):
        emu.input.keypad_add_key(keymask(Keys.KEY_RIGHT))
    elif key == "z":
        emu.input.keypad_add_key(keymask(Keys.KEY_B))
    elif key == "x":
        emu.input.keypad_add_key(keymask(Keys.KEY_A))


def on_key_release(widget, event):
    global emu, current_input
    assert emu is not None
    key = Gdk.keyval_name(event.keyval).lower()

    if key in ("w", "up"):
        emu.input.keypad_rm_key(keymask(Keys.KEY_UP))
    elif key in ("s", "down"):
        emu.input.keypad_rm_key(keymask(Keys.KEY_DOWN))
    elif key in ("a", "left"):
        emu.input.keypad_rm_key(keymask(Keys.KEY_LEFT))
    elif key in ("d", "right"):
        emu.input.keypad_rm_key(keymask(Keys.KEY_RIGHT))
    elif key == "z":
        emu.input.keypad_rm_key(keymask(Keys.KEY_B))
    elif key == "x":
        emu.input.keypad_rm_key(keymask(Keys.KEY_A))


# --- Drawing ---
def on_draw_main(widget: Gtk.DrawingArea, ctx: cairo.Context):
    global renderer
    assert renderer is not None

    # Draw emulator screen
    renderer.screen(SCREEN_WIDTH, SCREEN_HEIGHT, ctx, 0)

    # Draw overlay points
    
    for key, _points in points.items():
        if key == "red":
            ctx.set_source_rgb(1, 0, 0)  # red
        elif key == "blue":
            ctx.set_source_rgb(0, 0, 1)  # blue
            
        for point in _points:
            ctx.new_sub_path()
            ctx.arc(point[0], point[1], dot_radius, 0, 2 * 3.14159)
    
    ctx.fill()

    return False


def on_configure_main(widget: Gtk.DrawingArea, *args):
    global renderer
    assert renderer is not None
    renderer.reshape(widget, 0)
    return True


# --- GTK Window ---
class EmulatorWindow(Gtk.Window):
    def __init__(self):
        super().__init__(title="DeSmuME with Overlay Dot")
        self.set_default_size(SCREEN_WIDTH, SCREEN_HEIGHT)

        drawing_area = Gtk.DrawingArea()
        drawing_area.set_size_request(SCREEN_WIDTH, SCREEN_HEIGHT)
        drawing_area.connect("draw", on_draw_main)
        drawing_area.connect("configure-event", on_configure_main)
        self.add(drawing_area)
        self.drawing_area = drawing_area

        self.connect("destroy", Gtk.main_quit)
        self.connect("key-press-event", on_key_press)
        self.connect("key-release-event", on_key_release)
        self.set_events(Gdk.EventMask.KEY_PRESS_MASK | Gdk.EventMask.KEY_RELEASE_MASK)


# --- Overlay points setter ---
def set_overlay_points(pts: list[list[float]], color="red"):
    global points
    points[color] = pts


# --- Callback worker thread ---
def callback_worker():
    while True:
        emu_instance = callback_queue.get()
        if emu_instance is None:
            break  # exit signal
        try:
            callback(emu_instance)
        except Exception as e:
            print("Callback error:", e)
        callback_queue.task_done()


# --- Main initialization ---
def init_desmume_with_overlay(rom_path: str, callback_fn):
    global emu, renderer, callback

    callback = callback_fn  # store global callback reference

    # Initialize emulator
    emu = DeSmuME()
    emu.open(rom_path)
    emu.savestate.load(1)
    emu.volume_set(0)  # mute

    # Setup renderer
    renderer = AbstractRenderer.impl(emu)
    renderer.init()

    # Create GTK window
    win = EmulatorWindow()
    win.show_all()

    # Start callback worker thread
    thread = threading.Thread(target=callback_worker, daemon=True)
    thread.start()

    # Timer for stepping emulator and queuing callback
    def tick():
        emu.input.keymask = current_input
        emu.cycle()  # step one frame
        win.drawing_area.queue_draw()  # trigger redraw
        callback_queue.put(emu)  # queue callback work
        return True

    GLib.timeout_add(16, tick)  # ~60 FPS

    # Ensure worker thread stops on window close
    win.connect("destroy", lambda w: callback_queue.put(None))

    Gtk.main()
