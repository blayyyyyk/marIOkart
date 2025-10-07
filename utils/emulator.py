
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
SCALE = 3
SCREEN_WIDTH, SCREEN_HEIGHT = 256, 192
current_input = 0


# Queue for callback processing in a separate thread
callback_queue = queue.Queue()

       # what the worker prepares


class DrawObject:
    def set_color(self, r: float, g: float, b: float):
        ...
    
    def draw(self, ctx: cairo.Context):
        ...
    

class Point(DrawObject):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.rgb = (255.0, 0.0, 0.0)
        
    def set_color(self, r: float, g: float, b: float):
        self.rgb = (r, g, b)
        
    def draw(self, ctx: cairo.Context):
        ctx.set_source_rgb(*self.rgb)
        stroke_width = dot_radius * self.z
        
        ctx.new_sub_path()
        ctx.arc(self.x, self.y, stroke_width, 0, 2 * 3.14159)
        ctx.fill()
        
class Line(DrawObject):
    def __init__(self, p1: Point, p2: Point, stroke_width: int | None = None):
        self.p1 = p1
        self.p2 = p2
        self.stroke_width = stroke_width
        x1 = self.p1.x
        x2 = self.p2.x
        y1 = self.p1.y
        y2 = self.p2.y
        self.grad = cairo.LinearGradient(x1, y1, x2, y2)
        r1, g1, b1 = self.p1.rgb
        r2, g2, b2 = self.p2.rgb
        self.grad.add_color_stop_rgb(0, r1, g1, b1)
        self.grad.add_color_stop_rgb(1, r2, g2, b2)
        
    def draw(self, ctx: cairo.Context):
        if self.stroke_width is None:
            z = (self.p1.z + self.p2.z) / 2
            ctx.set_line_width(dot_radius * z)
        else:
            ctx.set_line_width(self.stroke_width)
        ctx.set_source(self.grad)
        ctx.new_sub_path()
        ctx.move_to(self.p1.x, self.p1.y)
        ctx.line_to(self.p2.x, self.p2.y)
        ctx.stroke()
        
class Triangle(DrawObject):
    def __init__(self, p1: Point, p2: Point, p3: Point, stroke_width: int | None = None):
        self.l1 = Line(p1, p2, stroke_width=stroke_width)
        self.l2 = Line(p2, p3, stroke_width=stroke_width)
        self.l3 = Line(p3, p1, stroke_width=stroke_width)
        
    def draw(self, ctx: cairo.Context):
        self.l1.draw(ctx)
        self.l2.draw(ctx)
        self.l3.draw(ctx)
        
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
            
    def add_lines(self, pts1: list[list[float]], pts2: list[list[float]], color: tuple[float, float, float], stroke_width: int | None = None):
        for p1, p2 in zip(pts1, pts2):
            line_obj = Line(
                Point(*p1),
                Point(*p2),
                stroke_width=stroke_width
            )
            line_obj.set_color(*color)
            self.add_obj(line_obj)
            
            
    def add_triangles(self, pts1: list[list[float]], pts2: list[list[float]], pts3: list[list[float]], color: tuple[float, float, float]):
        for p1, p2, p3 in zip(pts1, pts2, pts3):
            point_1 = Point(*p1)
            point_1.set_color(*color)
            point_2 = Point(*p2)
            point_2.set_color(*color)
            point_3 = Point(*p3)
            point_3.set_color(*color)
            tri_obj = Triangle(
                point_1,
                point_2,
                point_3,
                stroke_width=2, 
            )
            self.add_obj(tri_obj)
            
    
        
    def clear(self):
        self.draw_objs = []
        
    def draw(self, ctx: cairo.Context):
        for draw_obj in self.draw_objs:
            draw_obj.draw(ctx)
        
scene_lock = threading.Lock()
scene_current = Scene()   # what GTK renders
scene_next = None  


input_lock = threading.Lock()
input_state = set()  # holds active keys (e.g. {"up", "left", "a"})

def on_key_press(widget, event):
    key = Gdk.keyval_name(event.keyval).lower()
    with input_lock:
        input_state.add(key)
    return True

def on_key_release(widget, event):
    key = Gdk.keyval_name(event.keyval).lower()
    with input_lock:
        input_state.discard(key)
    return True

def apply_input_state(emu: DeSmuME, pressed):
    emu.input.keypad_update(0)

    keymap = {
        "w": Keys.KEY_UP, "up": Keys.KEY_UP,
        "s": Keys.KEY_DOWN, "down": Keys.KEY_DOWN,
        "a": Keys.KEY_LEFT, "left": Keys.KEY_LEFT,
        "d": Keys.KEY_RIGHT, "right": Keys.KEY_RIGHT,
        "z": Keys.KEY_B,
        "x": Keys.KEY_A,
        "8": Keys.KEY_START,
        "9": Keys.KEY_SELECT,
    }

    for k in pressed:
        if k in keymap:
            emu.input.keypad_add_key(keymask(keymap[k]))

def on_draw_main(widget: Gtk.DrawingArea, ctx: cairo.Context):
    global renderer, scene_current, scene_lock
    assert renderer is not None
    
    # Scale the whole drawing context
    ctx.scale(SCALE, SCALE)
    
    # Draw emulator screen
    renderer.screen(SCREEN_WIDTH, SCREEN_HEIGHT, ctx, 0)
    with scene_lock:
        scene_current.draw(ctx)
    
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
        drawing_area.set_size_request(SCREEN_WIDTH * SCALE, SCREEN_HEIGHT * SCALE)
        drawing_area.connect("draw", on_draw_main)
        drawing_area.connect("configure-event", on_configure_main)
        self.add(drawing_area)
        self.drawing_area = drawing_area

        self.connect("destroy", Gtk.main_quit)
        self.connect("key-press-event", on_key_press)
        self.connect("key-release-event", on_key_release)
        self.set_events(Gdk.EventMask.KEY_PRESS_MASK | Gdk.EventMask.KEY_RELEASE_MASK)


def draw_points(scene: Scene, pts: list[list[float]], color: tuple[float, float, float]):
    scene.add_points(pts, color)
    
def draw_triangles(scene: Scene, pts1: list[list[float]], pts2: list[list[float]], pts3: list[list[float]], color: tuple[float, float, float]):
    scene.add_triangles(pts1, pts2, pts3, color)

def callback_worker():
    global scene_next
    while True:
        emu_instance = callback_queue.get()
        if emu_instance is None:
            break  # exit signal
        try:
            new_scene = Scene()
            callback(emu_instance, new_scene)
            with scene_lock:
                scene_next = new_scene
        except Exception as e:
            raise RuntimeError("Custom thread failure")
        callback_queue.task_done()
            

def init_desmume_with_overlay(rom_path: str, callback_fn, init_fn):
    global emu, renderer, callback

    callback = callback_fn  # store global callback reference

    # Initialize emulator
    emu = DeSmuME()
    emu.open(rom_path)
    emu.savestate.load(3)
    emu.volume_set(0)  # mute
    init_fn(emu)

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
        global scene_current, scene_next
        assert emu is not None
        emu.cycle()
        
        with input_lock:
            pressed = set(input_state)
        
        apply_input_state(emu, pressed)
    
        # If worker produced a new scene, swap it in safely
        with scene_lock:
            if scene_next is not None:
                scene_current = scene_next
                scene_next = None
    
        win.drawing_area.queue_draw()  # render current scene
        callback_queue.put(emu)        # queue next computation
        return True

    GLib.timeout_add(16, tick)  # ~60 FPS

    # Ensure worker thread stops on window close
    win.connect("destroy", lambda w: callback_queue.put(None))

    Gtk.main()
