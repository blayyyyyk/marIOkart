from desmume.emulator import DeSmuME
from desmume.controls import Keys, keymask
from desmume.frontend.gtk_drawing_area_desmume import AbstractRenderer
from src.visualization.draw import consume_draw_stack
import cairo
from pynput import keyboard
import numpy as np

# GTK Imports
import gi

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gtk, Gdk, GLib

# ----------------------------
# CONSTANTS
# ----------------------------
SCALE = 3
SCREEN_WIDTH, SCREEN_HEIGHT = 256, 192

# ----------------------------
# GLOBAL VARIABLES
# ----------------------------
input_state = set()
renderer = None

# ----------------------------
# KEY MAPPING
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


# ----------------------------
# ASYNC KEYBOARD HANDLER
# ----------------------------
def start_keyboard_listener():
    """Starts a non-blocking keyboard listener in a separate thread."""

    def on_press(key):
        try:
            name = key.char.lower() if hasattr(key, "char") else key.name
            if name in KEY_MAP:
                input_state.add(name)
        except Exception:
            pass

    def on_release(key):
        try:
            name = key.char.lower() if hasattr(key, "char") else key.name
            if name in KEY_MAP:
                input_state.discard(name)
        except Exception:
            pass

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.daemon = True
    listener.start()
    return listener


# ----------------------------
# DRAW CALLBACK
# ----------------------------
overlay_surface_cache: cairo.Surface | None = None


def on_draw_memoryview(
    buf: memoryview, width: int, height: int, scale_x: float, scale_y: float
) -> np.ndarray:
    """Scale an emulator RGBX memoryview and return a scaled NumPy RGBA array."""
    stride = width * 4  # RGBX → 4 bytes per pixel
    # Wrap input buffer in a Cairo surface (no copy)
    src_surface = cairo.ImageSurface.create_for_data(
        buf, cairo.FORMAT_RGB24, width, height, stride
    )

    # Compute new scaled size
    new_w = int(width * scale_x)
    new_h = int(height * scale_y)

    # Create destination surface for scaled result
    dest_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, new_w, new_h)
    ctx = cairo.Context(dest_surface)

    # Apply scaling and draw the source surface into the new one
    ctx.scale(scale_x, scale_y)
    
    # apply overlay
    composited = apply_overlay_to_surface(src_surface)
    
    ctx.set_source_surface(composited, 0, 0)
    ctx.paint()

    # Extract raw BGRA pixel data → NumPy array
    buf_out = dest_surface.get_data()
    arr = np.ndarray((new_h, new_w, 4), np.uint8, buffer=buf_out)
    return arr


def apply_overlay_to_surface(src_surface: cairo.ImageSurface) -> cairo.Surface:
    """Return a new Cairo surface with overlay applied on top of the input surface."""
    global overlay_surface_cache

    width = src_surface.get_width()
    height = src_surface.get_height()

    # Create overlay surface (transparent, same size)
    overlay_surface = src_surface.create_similar(
        cairo.CONTENT_COLOR_ALPHA, width, height
    )
    overlay_surface.set_device_scale(*src_surface.get_device_scale())

    # Draw overlay contents
    overlay_ctx = cairo.Context(overlay_surface)
    overlay_ctx.set_operator(cairo.OPERATOR_CLEAR)
    overlay_ctx.paint()
    overlay_ctx.set_operator(cairo.OPERATOR_OVER)
    overlay_ctx.set_antialias(cairo.ANTIALIAS_NONE)

    num_calls = consume_draw_stack(overlay_ctx)

    # Decide which surface to use (cached or freshly drawn)
    if num_calls == 0 and overlay_surface_cache is not None:
        overlay = overlay_surface_cache
    else:
        overlay_surface_cache = overlay_surface
        overlay = overlay_surface

    # Create new composited output surface
    result_surface = src_surface.create_similar(
        cairo.CONTENT_COLOR_ALPHA, width, height
    )
    result_surface.set_device_scale(*src_surface.get_device_scale())

    ctx = cairo.Context(result_surface)
    ctx.set_source_surface(src_surface, 0, 0)
    ctx.paint()
    ctx.set_source_surface(overlay, 0, 0)
    ctx.paint()

    return result_surface


def on_draw_main(widget: Gtk.DrawingArea, ctx: cairo.Context):
    global renderer
    if renderer is None:
        return False

    ctx.scale(SCALE, SCALE)
    screen_width = ctx.get_target().get_width()
    screen_height = ctx.get_target().get_height()

    # draw emulator frame
    renderer.screen(screen_width, screen_height, ctx, 0)
    src_surface = ctx.get_target()

    # apply overlay
    composited = apply_overlay_to_surface(src_surface)

    # paint the final result to GTK window
    ctx.set_source_surface(composited, 0, 0)
    ctx.paint()

    return False


def on_configure_main(widget: Gtk.DrawingArea, *args):
    global renderer
    if renderer:
        renderer.reshape(widget, 0)
    return True


# ----------------------------
# GTK WINDOW
# ----------------------------
class EmulatorWindow(Gtk.Window):
    def __init__(self, emu: DeSmuME):
        # Init the renderer on window creation
        global renderer
        renderer = AbstractRenderer.impl(emu)
        renderer.init()

        super().__init__(title="MarI/O Kart")
        self.set_default_size(SCREEN_WIDTH, SCREEN_HEIGHT)
        drawing_area = Gtk.DrawingArea()
        drawing_area.set_size_request(SCREEN_WIDTH * SCALE, SCREEN_HEIGHT * SCALE)
        drawing_area.connect("draw", on_draw_main)
        drawing_area.connect("configure-event", on_configure_main)
        self.add(drawing_area)
        self.drawing_area = drawing_area
        self.connect("destroy", Gtk.main_quit)
        self.set_events(Gdk.EventMask.KEY_PRESS_MASK | Gdk.EventMask.KEY_RELEASE_MASK)

    def start(self, func, queue):
        # run at 60fps, non-blocking
        GLib.timeout_add(16, func)
        Gtk.main()
        self.connect("destroy", lambda w: queue.put(None))

    def kill(self):
        Gtk.main_quit()
