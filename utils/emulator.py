
import os
import threading
import queue
# --- Environment setup for Homebrew libraries ---
os.environ['PKG_CONFIG_PATH'] = "/opt/homebrew/lib/pkgconfig:$PKG_CONFIG_PATH"
os.environ['DYLD_FALLBACK_LIBRARY_PATH'] = "/opt/homebrew/lib:$DYLD_FALLBACK_LIBRARY_PATH"

import cairo
import gi
gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gtk, Gdk, GLib

from desmume.controls import Keys, keymask
from desmume.emulator import DeSmuME
from desmume.frontend.gtk_drawing_area_desmume import AbstractRenderer



# --- Global variables ---
emu = None
renderer = None
dot_radius = 5
SCREEN_WIDTH, SCREEN_HEIGHT = 256, 192
current_input = 0


# Queue for callback processing in a separate thread
callback_queue = queue.Queue()

class DrawRequest:
    def __init__(self, points: list[list[float]], color: tuple[float, float, float], depth: bool | None = False):
        self.points = points
        self.color = color
        self.depth = depth
    

points: list[DrawRequest] = []


class DrawObject:
    def set_color(self, r: float, g: float, b: float):
        ...
    
    def draw(self, ctx: cairo.Context):
        ...
    

class Point(DrawObject):
    def __init__(self, x, y, z=None):
        self.x = x
        self.y = y
        self.z = z if z is not None else 1
        self.rgb = (255.0, 0.0, 0.0)
        
    def set_color(self, r: float, g: float, b: float):
        self.rgb = (r, g, b)
        
    def draw(self, ctx: cairo.Context):
        ctx.set_source_rgb(*self.rgb)
        ctx.new_sub_path()
        ctx.arc(self.x, self.y, dot_radius * self.z, 0, 2 * 3.14159)
        ctx.fill()
        
class Line(DrawObject):
    def __init__(self, p1: Point, p2: Point, stroke_width: int):
        self.p1 = p1
        self.p2 = p2
        self.stroke_width = stroke_width
        
    def set_color(self, r: float, g: float, b: float):
        self.p1.set_color(r, g, b)
        
    @property
    def rgb(self):
        return self.p1.rgb
        
    def draw(self, ctx: cairo.Context):
        ctx.set_line_width(self.width)
        ctx.set_source_rgb(*self.rgb)
        ctx.new_sub_path()
        ctx.move_to(self.p1.x, self.p1.y)
        ctx.line_to(self.p2.x, self.p2.y)
        ctx.stroke()
        
class Triangle(DrawObject):
    def __init__(self, p1: Point, p2: Point, p3: Point, stroke_width: int, fill: bool):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.stroke_width = stroke_width
        self.fill = fill
        
        
    def set_color(self, r: float, g: float, b: float):
        self.p1.set_color(r, g, b)
        
    @property
    def rgb(self):
        return self.p1.rgb
        
    def draw(self, ctx: cairo.Context):
        ctx.set_line_width(self.stroke_width)
        ctx.set_source_rgb(*self.rgb)
        ctx.new_sub_path()
        ctx.move_to(self.p1.x, self.p1.y)
        ctx.line_to(self.p2.x, self.p2.y)
        ctx.line_to(self.p3.x, self.p3.y)
        ctx.close_path()
        if self.fill:
            ctx.fill()
        else:
            ctx.stroke()
        
class Scene:
    def __init__(self):
        self.draw_objs: list[DrawObject] = []
        
    def add_obj(self, obj: DrawObject):
        self.draw_objs.append(obj)
        
    def add_points(self, pts: list[list[float]], color: tuple[float, float, float]):
        for pt in pts:
            pt_obj = Point(*pt)
            pt_obj.set_color(*color)
            self.add_obj(pt_obj)
            
    def add_lines(self, pts1: list[list[float]], pts2: list[list[float]], color: tuple[float, float, float]):
        for p1, p2 in zip(pts1, pts2):
            line_obj = Line(
                Point(*p1),
                Point(*p2),
                stroke_width=5
            )
            line_obj.set_color(*color)
            self.add_obj(line_obj)
            
            
    def add_triangles(self, pts1: list[list[float]], pts2: list[list[float]], pts3: list[list[float]], color: tuple[float, float, float]):
        for p1, p2, p3 in zip(pts1, pts2, pts3):
            tri_obj = Triangle(
                Point(*p1), 
                Point(*p2), 
                Point(*p3), 
                stroke_width=5, 
                fill=False
            )
            tri_obj.set_color(*color)
            self.add_obj(tri_obj)
            
        
    def clear(self):
        self.draw_objs = []
        
    def draw(self, ctx: cairo.Context):
        for draw_obj in self.draw_objs:
            draw_obj.draw(ctx)
        
scene = Scene()

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
    global emu, current_input, points
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


def on_draw_main(widget: Gtk.DrawingArea, ctx: cairo.Context):
    global renderer
    assert renderer is not None
    # Draw emulator screen
    renderer.screen(SCREEN_WIDTH, SCREEN_HEIGHT, ctx, 0)
    scene.draw(ctx)
    scene.clear()
    return False
    

def on_configure_main(widget: Gtk.DrawingArea, *args):
    global renderer
    assert renderer is not None
    renderer.reshape(widget, 0)
    return True


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


def draw_checkpoints(pts: list[list[float]], color: tuple[float, float, float]):
    global scene
    scene.add_points(pts, color)
    
def draw_collisions(pts1: list[list[float]], pts2: list[list[float]], pts3: list[list[float]], color: tuple[float, float, float]):
    global scene
    scene.add_triangles(pts1, pts2, pts3, color)

def callback_worker():
    while True:
        emu_instance = callback_queue.get()
        if emu_instance is None:
            break  # exit signal
        try:
            callback(emu_instance)
        except Exception as e:
            raise RuntimeError("Custom thread failure")
        callback_queue.task_done()
            

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
        assert emu is not None
        emu.cycle()  # step one frame
        win.drawing_area.queue_draw()  # trigger redraw
        
        callback_queue.put(emu)  # queue callback work
        return True

    GLib.timeout_add(16, tick)  # ~60 FPS

    # Ensure worker thread stops on window close
    win.connect("destroy", lambda w: callback_queue.put(None))

    Gtk.main()
