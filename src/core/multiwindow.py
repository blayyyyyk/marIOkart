import numpy as np
import cairo

import gi
gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gtk, Gdk, GLib

renderer = None
overlay_surface_cache = None

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
            ctx.get_source().set_filter(cairo.Filter.NEAREST)  # Disable smoothing
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