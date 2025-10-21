from desmume.emulator import DeSmuME
from desmume.controls import Keys, keymask
from desmume.frontend.gtk_drawing_area_desmume import AbstractRenderer
from src.visualization.draw import consume_draw_stack
import cairo, numpy as np, math, time, os
from pynput import keyboard
from multiprocessing.shared_memory import SharedMemory
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
overlay_surface_cache = None


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


# ============================================================
#  Overlay-Composited Offscreen Draw
# ============================================================
def on_draw_memoryview(
    buf: memoryview, width: int, height: int, scale_x: float, scale_y: float
) -> np.ndarray:
    """Scale emulator RGBX frame and apply overlay → NumPy RGBA array."""
    stride = width * 4
    src_surface = cairo.ImageSurface.create_for_data(
        buf, cairo.FORMAT_RGB24, width, height, stride
    )
    new_w, new_h = int(width * scale_x), int(height * scale_y)
    dest_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, new_w, new_h)
    ctx = cairo.Context(dest_surface)
    ctx.scale(scale_x, scale_y)

    # Compose overlays
    composited = apply_overlay_to_surface(src_surface)
    ctx.set_source_surface(composited, 0, 0)
    ctx.paint()

    buf_out = dest_surface.get_data()
    arr = np.ndarray((new_h, new_w, 4), np.uint8, buffer=buf_out)
    return arr


def apply_overlay_to_surface(src_surface: cairo.ImageSurface) -> cairo.Surface:
    """Return new Cairo surface with overlay applied."""
    global overlay_surface_cache
    width, height = src_surface.get_width(), src_surface.get_height()

    overlay_surface = src_surface.create_similar(
        cairo.CONTENT_COLOR_ALPHA, width, height
    )
    overlay_surface.set_device_scale(*src_surface.get_device_scale())

    overlay_ctx = cairo.Context(overlay_surface)
    overlay_ctx.set_operator(cairo.OPERATOR_CLEAR)
    overlay_ctx.paint()
    overlay_ctx.set_operator(cairo.OPERATOR_OVER)
    overlay_ctx.set_antialias(cairo.ANTIALIAS_NONE)

    num_calls = consume_draw_stack(overlay_ctx)

    if num_calls == 0 and overlay_surface_cache is not None:
        overlay = overlay_surface_cache
    else:
        overlay_surface_cache = overlay_surface
        overlay = overlay_surface

    result_surface = src_surface.create_similar(
        cairo.CONTENT_COLOR_ALPHA, width, height
    )
    ctx = cairo.Context(result_surface)
    ctx.set_source_surface(src_surface, 0, 0)
    ctx.paint()
    ctx.set_source_surface(overlay, 0, 0)
    ctx.paint()
    return result_surface


# ============================================================
#  Multi-Process Display Window (shared memory compositor)
# ============================================================

class SharedEmulatorWindow(Gtk.Window):
    def __init__(self, width: int, height: int, n_cols: int, n_rows: int, renderer: AbstractRenderer, shm_names: list[str]):
        super().__init__(title="MarI/O Kart — Multi Display")
        self.connect("destroy", self._on_destroy)
        self.set_default_size(width, height)

        self.width = width
        self.height = height
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.renderer = renderer
        # Hold references so their buffers remain valid:
        self._shms = [SharedMemory(name=name) for name in shm_names]
        self.frames = [
            np.ndarray((SCREEN_HEIGHT, SCREEN_WIDTH, 4), dtype=np.uint8, buffer=shm.buf)
            for shm in self._shms
        ]
        
        self.area = Gtk.DrawingArea()
        self.area.connect("draw", self.on_draw)
        self.add(self.area)
        
        GLib.timeout_add(33, self.refresh)

    def on_draw(self, widget: Gtk.DrawingArea, ctx: cairo.Context):
        import math
        scale = 1.0
        tile_w = self.width / self.n_cols
        tile_h = tile_w * (SCREEN_HEIGHT / SCREEN_WIDTH)

        for i, frame in enumerate(self.frames):
            r, c = divmod(i, self.n_cols)
            
            surface = cairo.ImageSurface.create_for_data(
                frame, cairo.FORMAT_RGB24, SCREEN_WIDTH, SCREEN_HEIGHT
            )
            
            ctx.save()
            ctx.translate(c * tile_w, r * tile_h)
            ctx.scale(tile_w / SCREEN_WIDTH, tile_h / SCREEN_HEIGHT)
            ctx.set_source_surface(surface, 0, 0)
            ctx.paint()
            ctx.restore()
            
    
            
    def refresh(self):
        self.area.queue_draw()
        return True
        
    def _on_destroy(self, *_):
        # Close (do not unlink) — unlink after procs finish in the trainer
        for shm in self._shms:
            try:
                shm.close()
            except Exception:
                pass

# ============================================================
#  Standard Single Emulator GTK Window (for interactive mode)
# ============================================================

class EmulatorWindow(Gtk.Window):
    """Single-emulator interactive window with overlay support."""
    def __init__(self, emu: DeSmuME):
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
        GLib.timeout_add(16, func)
        Gtk.main()
        self.connect("destroy", lambda w: queue.put(None))

    def kill(self):
        Gtk.main_quit()


# ============================================================
#  GTK Draw Helpers (used by EmulatorWindow)
# ============================================================

def on_draw_main(widget: Gtk.DrawingArea, ctx: cairo.Context):
    global renderer, overlay_surface_cache
    if renderer is None:
        return False

    ctx.scale(SCALE, SCALE)
    renderer.screen(SCREEN_WIDTH, SCREEN_HEIGHT, ctx, 0)
    src_surface = ctx.get_target()

    overlay_surface = src_surface.create_similar(
        cairo.CONTENT_COLOR_ALPHA, SCREEN_WIDTH, SCREEN_HEIGHT
    )
    overlay_surface.set_device_scale(*src_surface.get_device_scale())

    overlay_ctx = cairo.Context(overlay_surface)
    overlay_ctx.set_operator(cairo.OPERATOR_CLEAR)
    overlay_ctx.paint()
    overlay_ctx.set_operator(cairo.OPERATOR_OVER)
    overlay_ctx.set_antialias(cairo.ANTIALIAS_NONE)

    num_calls = consume_draw_stack(overlay_ctx)
    if num_calls == 0 and overlay_surface_cache is not None:
        to_paint = overlay_surface_cache
    else:
        overlay_surface_cache = overlay_surface
        to_paint = overlay_surface

    pattern = cairo.SurfacePattern(to_paint)
    pattern.set_filter(cairo.FILTER_NEAREST)
    ctx.set_source(pattern)
    ctx.set_operator(cairo.OPERATOR_OVER)
    ctx.paint()
    return False


def on_configure_main(widget: Gtk.DrawingArea, *args):
    global renderer
    if renderer:
        renderer.reshape(widget, 0)
    return True
