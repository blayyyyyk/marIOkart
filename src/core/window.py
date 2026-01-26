from core.emulator import MarioKart, MarioKart_Memory
from desmume.controls import Keys, keymask
from desmume.frontend.gtk_drawing_area_desmume import AbstractRenderer
from src.visualization.draw import consume_draw_stack, _draw_lines, _draw_triangles
import cairo, numpy as np, math, time, os
from pynput import keyboard
from multiprocessing.shared_memory import SharedMemory
from typing import Callable, Any, TypeVar, Generic, TypedDict
from queue import Queue
import threading
import queue
from abc import ABC, abstractmethod
import gi
import torch
from src.utils.vector import (
    pairwise_distances_cross,
    intersect_ray_line_2d,
    sample_circular_sweep,
    triangle_raycast_batch,
    sample_cone,
    triangle_altitude,
    generate_plane_vectors,
    generate_driver_rays
)
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

COLOR_MAP = [
    # --- DRIVEABLE SURFACES (Grays/Whites) ---
    [128, 128, 128],  # 0:  Road (Standard Gray)
    [200, 230, 255],  # 1:  Slippery Road (Icy Blue-White)
    
    # --- OFFROAD (Greens/Browns) ---
    [34,  139, 34],   # 2:  Weak Offroad (Forest Green - Grass)
    [139, 69,  19],   # 3:  Offroad (Saddle Brown - Mud/Dirt)
    
    # --- TECHNICAL (Purples) ---
    [238, 130, 238],  # 4:  Sound Trigger (Violet)
    
    # --- HEAVY OFFROAD ---
    [80,  50,  20],   # 5:  Heavy Offroad (Dark Brown - Deep Mud)
    
    # --- SLIPPERY VARIANTS ---
    [175, 238, 238],  # 6:  Slippery Road 2 (Pale Turquoise)
    
    # --- BOOSTS (Oranges) ---
    [255, 140, 0],    # 7:  Boost Panel (Dark Orange)
    
    # --- WALLS (Reds) ---
    [255, 0,   0],    # 8:  Wall (Pure Red)
    [255, 105, 180],  # 9:  Invisible Wall (Hot Pink - distinct from normal wall)
    
    # --- BOUNDARIES (Blacks/Darks) ---
    [0,   0,   0],    # 10: Out of Bounds (Black)
    [25,  25,  112],  # 11: Fall Boundary (Midnight Blue - Abyss)
    
    # --- JUMPS (Yellows) ---
    [255, 255, 0],    # 12: Jump Pad (Yellow)
    
    # --- AI/DRIVER LOGIC (Tinted Grays) ---
    [169, 169, 169],  # 13: Road (no drivers) (Dark Gray)
    [139, 0,   0],    # 14: Wall (no drivers) (Dark Red)
    
    # --- MECHANICS (Metals/Indigos) ---
    [75,  0,   130],  # 15: Cannon Activator (Indigo)
    [205, 92,  92],   # 16: Edge Wall (Indian Red)
    
    # --- WATER ---
    [0,   0,   255],  # 17: Falls Water (Pure Blue)
    
    # --- BOOST VARIANT ---
    [255, 69,  0],    # 18: Boost Pad w/ Min Speed (Red-Orange)
    
    # --- SPECIAL ROADS ---
    [192, 192, 192],  # 19: Loop Road (Silver)
    [255, 215, 0],    # 20: Special Road (Gold - e.g., Rainbow Road segments)
    
    # --- WALL VARIANT ---
    [128, 0,   0],    # 21: Wall 3 (Maroon)
    
    # --- RECALC ---
    [0,   255, 0]     # 22: Force Recalc (Lime Green - Debug visual)
]


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

# ============================================================
#  Standard Single Emulator GTK Window (for interactive mode)
# ============================================================

T = TypeVar("T")
class AsyncOverlay(Gtk.DrawingArea, Generic[T]):
    def __init__(self, emu, device, refresh_rate = 60.0):
        super(Gtk.DrawingArea, self).__init__()
        self.device = device
        self.refresh_interval = 1.0 / refresh_rate
        self.memory = MarioKart_Memory(emu, device=device)
        self.set_app_paintable(True) # ignore
        self.connect('draw', self._on_draw_wrapper)

        self._running = False
        self.lock = threading.Lock()
        self.rendering_state: T | None = None # cache
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
        # 1. READ STATE SAFELY
        with self.lock:
            state = self.rendering_state
            
        if state is None:
            return False

        # 2. RENDER
        ctx.scale(SCALE, SCALE) # Assuming global SCALE

        ctx.save()

        ctx.rectangle(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
        ctx.clip()

        self.draw_overlay(ctx, state)

        ctx.restore()
        return False
    
    def compute(self, state: T | None) -> T:
        ...

    def draw_overlay(self, ctx: cairo.Context, state: T):
        pass

class CheckpointOverlayI(TypedDict):
    layer_1: np.ndarray
    layer_2: np.ndarray

class CheckpointOverlay(AsyncOverlay[CheckpointOverlayI]):
    def compute(self, state) -> CheckpointOverlayI:
        if state is None:
            return {
                'layer_1': np.zeros((2, 3)),
                'layer_2': np.zeros((2, 3))
            }

        device = self.device
        position = self.memory.driver.position
        position = position.to(device)
        
        # Checkpoint boundary
        checkpoint = self.memory.checkpoint_info()['next_checkpoint_pos']
        checkpoint = checkpoint.to(device)
        checkpoint[:, 1] -= self.memory.camera.targetElevation / (1 << 12)

        proj = self.memory.project_to_screen(checkpoint, normalize_depth=True)
        screen_space = proj['screen']
        depth_mask = proj['mask']
        screen_space = screen_space[depth_mask]
        screen_space = screen_space.detach().cpu().numpy()

        if screen_space.shape[0] == 2:
            assert isinstance(state, dict)
            state['layer_1'] = screen_space
        
        # Path to checkpoint
        screen_space = None
        intersect = self.memory.read_facing_point_checkpoint(device=device)
        points = torch.stack([intersect, position], dim=0)
        proj = self.memory.project_to_screen(points, normalize_depth=True)
        screen_space = proj['screen']
        depth_mask = proj['mask']
        depth_mask[1] = True
        screen_space = screen_space[depth_mask]
        screen_space = screen_space.detach().cpu().numpy()
        if screen_space.shape[0] < 2:
            return state
        
        
        state['layer_2'] = screen_space
        return state

    def draw_overlay(self, ctx, state: CheckpointOverlayI):
        layer_1 = state['layer_1']
        p1, p2 = layer_1[0], layer_1[1]

        layer_2 = state['layer_2']
        p3, p4 = layer_2[0], layer_2[1]
        
        p1 = np.stack([p1, p3], axis=0)
        p2 = np.stack([p2, p4], axis=0)
        colors = np.array([0.0, 1.0, 0.0])
        _draw_lines(ctx, p1, p2, colors=colors, stroke_width_scale=1.0)


CollisionOverlayI = tuple[np.ndarray, np.ndarray]
class CollisionOverlay(AsyncOverlay[CollisionOverlayI]):
    def compute(self, state):
        if state is None:
            return np.zeros((3, 3, 3)), np.zeros((3, 3))

        v1 = self.memory.collision_data['v1'].to(self.device)
        v2 = self.memory.collision_data['v2'].to(self.device)
        v3 = self.memory.collision_data['v3'].to(self.device)
        group = torch.cat([v1, v2, v3], dim=0)
        v_proj = self.memory.project_to_screen(group, normalize_depth=True)
        
        # z filter
        m1, m2, m3 = v_proj['mask'].chunk(3, dim=0)
        z_mask = m1 & m2 & m3
        tri = v_proj['screen'].view(3, v1.shape[0], -1)
        tri = tri[:, z_mask]

        if tri.shape[0] == 0:
            return self.rendering_state

        # material color
        color_map = torch.tensor(COLOR_MAP, dtype=torch.uint8, device=self.device)
        collision_type = self.memory.collision_data['prism_attribute']['collision_type']
        collision_type = collision_type[z_mask]
        floor_mask = self.memory.collision_data['prism_attribute']['is_floor'][z_mask]==1
        wall_mask = self.memory.collision_data['prism_attribute']['is_wall'][z_mask]!=1
        collision_type = collision_type[floor_mask & wall_mask]
        color_ids = torch.tensor(collision_type, dtype=torch.int32, device=self.device)
        colors = color_map[color_ids]
        #colors = colors[z_mask]
        #colors = torch.tensor([[0, 255, 128]], device=self.device)
        tri = tri[:, floor_mask & wall_mask]

        colors = colors.detach().cpu().numpy()
        tri = tri.detach().cpu().numpy()

        return tri, colors

    def draw_overlay(self, ctx, state):
        if not isinstance(state, tuple):
            return
        
        tri, colors = state
        v1, v2, v3 = np.unstack(tri)
        _draw_triangles(ctx, v1, v2, v3, colors)

def _min_max_scale(X):
    return (X - X.min()) / (X.max() - X.min())



class SensorOverlay(AsyncOverlay[tuple[np.ndarray, np.ndarray, np.ndarray]]):
    def compute(self, state):
        if state is None:
            return np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 3))

        P = self.memory.driver.position.to(self.device)
        P = P.unsqueeze(0)
        max_dist = 3000.0
        info = self.memory.obstacle_info(128, max_dist=max_dist, device=self.device)
        R, D = info['position'], info['distance']

        R_s = self.memory.project_to_screen(R, normalize_depth=True)['screen']
        P_s = self.memory.project_to_screen(P, normalize_depth=True)['screen']
        P_s = P_s.expand_as(R_s)
        P = P.expand_as(R)
        colors = torch.tensor([1., 0., 0.]).to(self.device)
        colors = colors.expand_as(P)
        weight = (D - D.mean()) / D.std()
        weight = weight.clamp(0, 1.0)
        colors = colors.clone()
        colors[:, 0] -= weight
        colors[:, 1] += weight

        return R_s.detach().cpu().numpy(), P_s.detach().cpu().numpy(), colors.detach().cpu().numpy()
    
    def draw_overlay(self, ctx, state):
        p1, p2, colors = state
        _draw_lines(ctx, p2, p1, colors=colors, stroke_width_scale=1.0)

class OrientationOverlay(AsyncOverlay[tuple[np.ndarray, np.ndarray]]):
    def compute(self, state):
        if state is None:
            return np.identity(3), np.zeros((3, 3))

        M = self.memory.driver.mainMtx.to(self.device)[:3, :] * 5.0 
        pos: torch.Tensor = self.memory.driver.position.to(self.device)

        P = pos.expand_as(M)
        M = self.memory.project_to_screen(M + P, normalize_depth=True)['screen']
        P = self.memory.project_to_screen(P, normalize_depth=True)['screen']
        state = M.detach().cpu().numpy(), P.detach().cpu().numpy()
        return state
    
    def draw_overlay(self, ctx, state):
        M, P = state
        colors = np.identity(3, dtype=np.float32)
        _draw_lines(ctx, M, P, colors)

class DriftOverlay(AsyncOverlay[tuple[np.ndarray, np.ndarray]]):
    def compute(self, state):
        if state is None:
            return np.zeros((1, 3)), np.zeros((1, 3))

        D = self.memory.driver.drivingDirection.unsqueeze(0) * 10.0
        P = self.memory.driver.position.unsqueeze(0)

        D = self.memory.project_to_screen(D + P, normalize_depth=True)['screen']
        P = self.memory.project_to_screen(P, normalize_depth=True)['screen']

        return D.detach().cpu().numpy(), P.detach().cpu().numpy()

    def draw_overlay(self, ctx, state):
        D, P = state
        colors = np.array([[0.9, 0.1, 0.9]])
        _draw_lines(ctx, D, P, colors)
        

class EmulatorWindow(Gtk.Window):
    def __init__(self, emu: MarioKart, overlays: list[type[AsyncOverlay]], device=None):
        super().__init__()
        
        # 1. State Flag
        self.show_sub_screen = True
        
        # Dimensions
        self.DS_WIDTH = 256
        self.DS_HEIGHT = 192
        self.TOTAL_WIDTH = self.DS_WIDTH * 2
        
        self.set_title("MarI/O Emulator Window")

        # Set initial size
        self.set_default_size(self.TOTAL_WIDTH * SCALE, self.DS_HEIGHT * SCALE)
        self.connect("destroy", self._on_window_destroy)

        self.renderer = AbstractRenderer.impl(emu)
        self.renderer.init()
        
        self.overlay_container = Gtk.Overlay()
        self.add(self.overlay_container)

        self.game_area = Gtk.DrawingArea()
        self.game_area.set_size_request(self.TOTAL_WIDTH * SCALE, self.DS_HEIGHT * SCALE)
        self.game_area.connect("draw", self.on_draw_main)
        self.game_area.connect("configure-event", self.on_configure_main)
        self.overlay_container.add(self.game_area)

        # Overlays
        self.overlays: dict[str, AsyncOverlay] = {}
        for overlay_cls in overlays:
            self.add_async_overlay(overlay_cls, emu, device=device, refresh_rate=60)

        self.show_all()

    # --- NEW METHOD ---
    def toggle_display_mode(self):
        """
        Toggles the visibility of the bottom (sub) screen.
        Resizes the window and prevents the backend from rendering the second screen.
        """
        self.show_sub_screen = not self.show_sub_screen
        
        # 1. Calculate Target Dimensions
        if self.show_sub_screen:
            # Showing both: Width is 2x
            target_width = self.TOTAL_WIDTH * SCALE
        else:
            # Showing top only: Width is 1x
            target_width = self.DS_WIDTH * SCALE
            
        target_height = self.DS_HEIGHT * SCALE

        # 2. Update DrawingArea constraints
        # We must reduce the 'request' size or GTK won't let the window shrink.
        self.game_area.set_size_request(target_width, target_height)
        
        # 3. Resize the Window
        self.resize(target_width, target_height)
        
        # 4. Queue Resize
        # This triggers 'configure-event', which updates our renderer buffers
        self.game_area.queue_resize()

    def add_async_overlay(self, overlay_cls: type[AsyncOverlay], emu, device, refresh_rate):
        name = overlay_cls.__name__
        overlay = overlay_cls(emu, device=device, refresh_rate=refresh_rate)
        overlay.set_can_focus(False)
        self.overlay_container.add_overlay(overlay)
        self.overlay_container.set_overlay_pass_through(overlay, True)
        overlay.show()
        self.overlays[name] = overlay

    def show_overlays(self):
        for overlay in self.overlays.values():
            overlay.start()

    @property
    def running(self):
        return all([overlay._running for overlay in self.overlays.values()])

    def _on_window_destroy(self, widget):
        self.on_window_destroy()

    def on_window_destroy(self):
        print("Shutting down...")
        for overlay in self.overlays.values():
            overlay.end()
        Gtk.main_quit()

    def on_draw_main(self, widget: Gtk.DrawingArea, ctx: cairo.Context, *args):
        ctx.scale(SCALE, SCALE)
        
        # 1. Always Draw Top Screen (Index 0)
        self.renderer.screen(self.DS_WIDTH * SCALE, self.DS_HEIGHT * SCALE, ctx, 0)
        
        # 2. Conditionally Draw Bottom Screen (Index 1)
        if self.show_sub_screen:
            ctx.save()
            ctx.translate(self.DS_WIDTH, 0) 
            self.renderer.screen(self.DS_WIDTH * SCALE, self.DS_HEIGHT * SCALE, ctx, 1)
            ctx.restore()
        
        return True

    def on_configure_main(self, widget: Gtk.DrawingArea, *args):
        # 1. Always allocate Top Screen
        self.renderer.reshape(widget, 0)
        
        # 2. Conditionally allocate Bottom Screen
        # If hidden, we stop calling reshape. This prevents the backend from 
        # resizing/reallocating the buffer for screen 1 during this event.
        if self.show_sub_screen:
            self.renderer.reshape(widget, 1)
            
        return True
        


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
