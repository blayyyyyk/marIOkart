import numpy as np, cairo
from typing import Literal, Optional, Generic, TypeVar
from desmume.emulator import SCREEN_WIDTH, SCREEN_HEIGHT
from desmume.frontend.gtk_drawing_area_desmume import AbstractRenderer
from src.core.emulator import MarioKart
from multiprocessing.shared_memory import SharedMemory
from src.core.emulator import MarioKart_Memory
import threading, time

import gi
gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gtk, Gdk, GLib

T = TypeVar("T")
class AsyncOverlay(Gtk.DrawingArea, Generic[T]):
    def __init__(self, emu, scale, device, refresh_rate=60.0, shm_name=None):
        super(Gtk.DrawingArea, self).__init__()
        self.device = device
        self.scale = scale
        self.refresh_interval = 1.0 / refresh_rate
        self.memory = MarioKart_Memory(emu, device=device)
        self.shm = None
        self.shm_buf = None
        if shm_name is not None:
            self.shm = SharedMemory(name=shm_name)
            self.shm_buf = self.shm.buf

        self.set_app_paintable(True)
        self.connect("draw", self._on_draw_wrapper)

        self._running = False
        self.lock = threading.Lock()
        self.rendering_state: T | None = None  # cache
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)

    def start(self):
        if not self._running:
            self._running = True
            self.thread.start()

    def end(self):
        self._running = False

    def _worker_loop(self):
        """Runs in background. Reads memory, does math."""
        while self._running:
            start_time = time.time()

            # 1. READ MEMORY & COMPUTE
            # We copy the result to a local var so we don't block the UI thread
            # during the actual heavy calculation.
            new_state = self.compute(self.rendering_state)

            # 2. UPDATE STATE SAFELY
            with self.lock:
                self.rendering_state = new_state

            # 3. REQUEST REDRAW (Thread-safe)
            # This tells GTK: "When you have a moment, call my draw function."
            GLib.idle_add(self.queue_draw)

            # Sleep to maintain target refresh rate for this specific overlay
            elapsed = time.time() - start_time
            sleep_time = max(0, self.refresh_interval - elapsed)
            time.sleep(sleep_time)

    def _on_draw_wrapper(self, widget, ctx):
        """Runs on Main UI Thread."""
        with self.lock:
            state = self.rendering_state

        if state is None:
            return False

        if self.shm_buf is not None:
            surf = cairo.ImageSurface.create_for_data(
                self.shm_buf, cairo.FORMAT_RGB24, SCREEN_WIDTH * 2, SCREEN_HEIGHT
            )
            _ctx = cairo.Context(surf)
        else:
            _ctx = ctx
            _ctx.scale(self.scale, self.scale)

        _ctx.save()
        _ctx.rectangle(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
        _ctx.clip()

        self.draw_overlay(_ctx, state)

        _ctx.restore()
        return False

    def compute(self, state: T | None) -> T: ...

    def draw_overlay(self, ctx: cairo.Context, state: T):
        pass

def draw_points(ctx: cairo.Context, pts: np.ndarray, colors: np.ndarray, radius_scale: float | np.ndarray):
    if isinstance(radius_scale, float):
        radius_scale = radius_scale * np.array(1)

    if pts.ndim == 1:
        pts = pts[None, :]

    if colors.ndim == 1:
        colors = colors[None, :]

    assert colors.shape[0] == 1 or colors.shape[0] == pts.shape[0]
    if colors.shape[0] == 1:
        colors = colors.repeat(pts.shape[0], axis=0)

    for (x, y, z), (r, g, b) in zip(pts, colors):
        ctx.set_source_rgb(r, g, b)
        ctx.arc(x, y, radius_scale * z, 0, 2 * np.pi)
        ctx.fill()

def draw_lines(ctx: cairo.Context, pts1: np.ndarray, pts2: np.ndarray, colors: np.ndarray, stroke_width_scale=1.0):
    if pts1.ndim == 1:
        pts1 = pts1[None, :]

    if pts2.ndim == 1:
        pts2 = pts2[None, :]

    assert pts2.shape[0] == pts1.shape[0], "All point arrays must have the same batch size"

    if colors.ndim == 1:
        colors = colors[None, :]

    assert colors.shape[0] == 1 or colors.shape[0] == pts1.shape[0]
    if colors.shape[0] == 1:
        colors = colors.repeat(pts1.shape[0], axis=0)

    for (p1, p2, (r, g, b)) in zip(pts1, pts2, colors):
        ctx.set_source_rgb(r, g, b)
        ctx.set_line_width(stroke_width_scale)
        ctx.move_to(*p1[:2])
        ctx.line_to(*p2[:2])
        ctx.stroke()

def draw_triangles(
    ctx: cairo.Context,
    pts1: np.ndarray,
    pts2: np.ndarray,
    pts3: np.ndarray,
    colors: np.ndarray
):
    n = pts1.shape[0]
    assert pts2.shape[0] == n and pts3.shape[0] == n, "All point arrays must have the same batch size"

    colors = np.asarray(colors)
    if colors.ndim == 1:
        colors = colors[None, :]
    assert colors.shape[1] == 3, "colors must have 3 channels (RGB)"
    if colors.shape[0] == 1:
        colors = np.repeat(colors, n, axis=0)
    else:
        assert colors.shape[0] == n, "colors must be [1,3] or [N,3]"

    l1 = np.concatenate([pts2, pts3, pts1], axis=0)  # p2->p3, p3->p1, p1->p2
    l2 = np.concatenate([pts3, pts1, pts2], axis=0)
    c3 = np.tile(colors, (3, 1))

    draw_lines(ctx, l1, l2, c3)  # assumes draw_lines accepts NumPy arrays

class EmulatorContainerImpl:
    width: int
    height: int
    scale: int

    def setup_container(
        self,
        width: int,
        height: int,
        orientation: Literal["horizontal", "vertical"],
        scale: int,
        id: Optional[int],
    ):
        self.width = width
        self.height = height
        self.total_width = self.width
        self.total_height = self.height
        self.scale = scale
        if orientation == "horizontal":
            self.total_width *= 2
        elif orientation == "vertical":
            self.total_height *= 2

        self.size = self.total_height * self.total_width * 8
        self.shm_buf = None
        self.shm_surface = None
        self.shm_name = None
        if id is None:
            return

        self.shm_name = f"display_{id}"
        self.shm = SharedMemory(name=self.shm_name, size=self.size)

        assert self.shm.buf
        self.shm_buf = self.shm.buf
        self.shm_surface = cairo.ImageSurface.create_for_data(
            self.shm_buf, cairo.FORMAT_RGB24, self.total_width, self.total_height
        )


def unpack_shm(name: str):
    shm = SharedMemory(name=name, size=SCREEN_WIDTH * SCREEN_HEIGHT * 8)
    arr = np.ndarray(
        shape=(SCREEN_WIDTH, SCREEN_HEIGHT, 3), dtype=np.uint8, buffer=shm.buf
    )
    return arr


class EmulatorWindowImpl(EmulatorContainerImpl):
    def setup_emulator(
        self,
        emu: MarioKart,
        overlays: list[type[AsyncOverlay]],
        width: int,
        height: int,
        orientation: Literal["horizontal", "vertical"],
        scale: int,
        id: Optional[int],
        device=None,
    ):
        self.setup_container(width, height, orientation, scale, id)
        self.show_sub_screen = True

        # Gtk Window Setup #
        self.set_title("MarI/O Emulator Window")
        self.set_default_size(
            self.total_width * self.scale, self.total_height * self.scale
        )
        self.connect("destroy", self._on_window_destroy)
        self._emu = emu
        self.renderer = AbstractRenderer.impl(emu)
        self.renderer.init()

        # Gtk Overlay Widgets Setup #
        self.overlay_container = Gtk.Overlay()
        self.add(self.overlay_container)
        self.game_area = Gtk.DrawingArea()
        self.game_area.set_size_request(
            self.total_width * self.scale, self.total_height * self.scale
        )
        self.game_area.connect("draw", self.on_draw_main)
        self.overlay_container.add(self.game_area)
        self.overlays: dict[str, AsyncOverlay] = {}

        for overlay_cls in overlays:
            self.add_async_overlay(
                overlay_cls,
                emu,
                scale,
                device=device,
                refresh_rate=60,
                shm_name=self.shm_name,
            )

    def detach_container(self):
        self.remove(self.overlay_container)
        return self.overlay_container

    def toggle_display_mode(self):
        """
        Toggles the visibility of the bottom (sub) screen.
        Resizes the window and prevents the backend from rendering the second screen.
        """
        self.show_sub_screen = not self.show_sub_screen
        if self.show_sub_screen:
            # Showing both: Width is 2x
            target_width, target_height = self.total_width, self.total_height
        else:
            # Showing top only: Width is 1x
            target_width, target_height = self.width, self.height

        target_width *= self.scale
        target_height *= self.scale
        self.game_area.set_size_request(target_width, target_height)
        self.resize(target_width, target_height)
        self.game_area.queue_resize()

    def add_async_overlay(
        self,
        overlay_cls: type[AsyncOverlay],
        emu,
        scale,
        device,
        refresh_rate,
        shm_name=None,
    ):
        name = overlay_cls.__name__
        overlay = overlay_cls(
            emu,
            scale,
            device=device,
            refresh_rate=refresh_rate,
            shm_name=shm_name,
        )
        overlay.set_can_focus(False)
        self.overlay_container.add_overlay(overlay)
        self.overlay_container.set_overlay_pass_through(overlay, True)
        self.overlays[name] = overlay

    def start_overlay_threads(self):
        for overlay in self.overlays.values():
            overlay.show()
            overlay.start()

    @property
    def running(self):
        return all([overlay._running for overlay in self.overlays.values()])

    def _on_window_destroy(self, widget):
        self.on_window_destroy()

    def on_window_destroy(self):
        for overlay in self.overlays.values():
            overlay.end()

        Gtk.main_quit()

    def on_draw_main(self, widget: Gtk.DrawingArea, ctx: cairo.Context, *args) -> bool:
        ctx.scale(self.scale, self.scale)

        # 1. Always Draw Top Screen (Index 0)
        target_width, target_height = self.width, self.height
        target_width *= self.scale
        target_height *= self.scale
        self.renderer.screen(target_width, target_height, ctx, 0)

        # 2. Conditionally Draw Bottom Screen (Index 1)
        if self.show_sub_screen:
            ctx.save()
            ctx.translate(
                self.total_width - self.width, self.total_height - self.height
            )
            self.renderer.screen(target_width, target_height, ctx, 1)
            ctx.restore()

        return True
