import multiprocessing
from src.core.emulator import MarioKart, MarioKart_Memory
from desmume.controls import Keys, keymask
from desmume.emulator import SCREEN_PIXEL_SIZE
from multiprocessing import Process
from src.display.overlay import AsyncOverlay
from src.display._impl import EmulatorWindowImpl
import cairo
from multiprocessing.shared_memory import SharedMemory
from typing import Literal, overload
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

class RealWindow(Gtk.Window, EmulatorWindowImpl):
    def __init__(self, *args, **kwargs):
        Gtk.Window.__init__(self)
        self.setup_emulator(*args, **kwargs)


class VirtualWindow(Gtk.OffscreenWindow, EmulatorWindowImpl):
    def __init__(self, *args, **kwargs):
        assert "id" in kwargs, "Virtual window must have an id"
        Gtk.OffscreenWindow.__init__(self)
        self.setup_emulator(*args, **kwargs)
        self.show_sub_screen = False
        self.id = kwargs["id"]

    def on_draw_main(self, widget: Gtk.DrawingArea, ctx: cairo.Context, *args):
        stop = SharedMemory(name="stop", size=1)
        assert stop.buf is not None
        if stop.buf[0] == 1:
            Gtk.main_quit()
            return False

        game_buf = self._emu.display_buffer_as_rgbx()
        surf_top = cairo.ImageSurface.create_for_data(
            game_buf[: SCREEN_PIXEL_SIZE * 4],
            cairo.FORMAT_RGB24,
            self.width,
            self.height,
        )

        assert self.shm_buf
        self.shm_surface = cairo.ImageSurface.create_for_data(
            self.shm_buf, cairo.FORMAT_RGB24, self.total_width, self.total_height
        )

        shm_ctx = cairo.Context(self.shm_surface)
        
        shm_ctx.set_source_surface(surf_top)
        shm_ctx.paint()

        if self.show_sub_screen:
            surf_bottom = cairo.ImageSurface.create_for_data(
                game_buf[SCREEN_PIXEL_SIZE * 4 :],
                cairo.FORMAT_RGB24,
                self.width,
                self.height,
            )

            shm_ctx.save()
            shm_ctx.translate(
                self.total_width - self.width, self.total_height - self.height
            )
            shm_ctx.set_source_surface(surf_bottom)
            shm_ctx.paint()
            shm_ctx.restore()

        return True

@overload
def EmulatorWindow(
    emu: MarioKart,
    overlays: list[type[AsyncOverlay]],
    width: int,
    height: int,
    orientation: Literal["horizontal", "vertical"],
    scale: int,
    virtual_screen: Literal[True],
    id: int = 0,
    device=None,
) -> VirtualWindow: ...


@overload
def EmulatorWindow(
    emu: MarioKart,
    overlays: list[type[AsyncOverlay]],
    width: int,
    height: int,
    orientation: Literal["horizontal", "vertical"],
    scale: int,
    virtual_screen: Literal[False] = False,
    id: int = 0,
    device=None,
) -> RealWindow: ...


def EmulatorWindow(
    emu: MarioKart,
    overlays: list[type[AsyncOverlay]],
    width: int,
    height: int,
    orientation: Literal["horizontal", "vertical"],
    scale: int,
    virtual_screen: bool = False,
    id: int = 0,
    device=None,
):
    if virtual_screen:
        return VirtualWindow(
            emu, overlays, width, height, orientation, scale, device=device, id=id
        )
    else:
        return RealWindow(
            emu, overlays, width, height, orientation, scale, device=device, id=None
        )






