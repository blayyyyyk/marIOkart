import multiprocessing
from src.core.emulator import MarioKart, MarioKart_Memory
from desmume.controls import Keys, keymask
from desmume.emulator import SCREEN_PIXEL_SIZE
from multiprocessing import Process
from desmume.frontend.gtk_drawing_area_desmume import AbstractRenderer
from src.visualization.draw import consume_draw_stack, _draw_lines, _draw_triangles
import cairo, numpy as np, math, time, os
from pynput import keyboard
from multiprocessing.shared_memory import SharedMemory
from typing import Callable, Any, Literal, TypeVar, Generic, TypedDict, Optional, Unpack, overload
from queue import Queue
import threading
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
        


class EmulatorContainerImpl:
    width: int
    height: int
    scale: int
    
    def setup_container(
        self,
        width: int,
        height: int,
        orientation: Literal['horizontal', 'vertical'],
        scale: int,
        id: Optional[int]
    ):
        self.width = width
        self.height = height
        self.total_width = self.width
        self.total_height = self.height
        self.scale = scale
        if orientation == 'horizontal':
            self.total_width *= 2
        elif orientation == 'vertical':
            self.total_height *= 2
        
        self.size = self.total_height * self.total_width * 8
        if id is None: return
        self.shm = SharedMemory(
            name=f"display_{id}",
            size=self.size
        )
        print(self.shm)
        
        
        assert self.shm.buf
        self.shm_buf = self.shm.buf
        self.shm_surface = cairo.ImageSurface.create_for_data(
            self.shm_buf, cairo.FORMAT_RGB24, self.total_width, self.total_height
        )
    
def unpack_shm(name: str):
    shm = SharedMemory(name=name, size=SCREEN_WIDTH*SCREEN_HEIGHT*8)
    arr = np.ndarray(shape=(SCREEN_WIDTH, SCREEN_HEIGHT, 3), dtype=np.uint8, buffer=shm.buf)
    return arr
    

class EmulatorWindowImpl(EmulatorContainerImpl):
    def setup_emulator(
        self, 
        emu: MarioKart, 
        overlays: list[type[AsyncOverlay]], 
        width: int,
        height: int,
        orientation: Literal['horizontal', 'vertical'],
        scale: int,
        id: Optional[int],
        device=None):
        self.setup_container(width, height, orientation, scale, id)
        self.show_sub_screen = True
        
        # Gtk Window Setup #
        self.set_title("MarI/O Emulator Window")
        self.set_default_size(self.total_width * self.scale, self.total_height * self.scale)
        self.connect("destroy", self._on_window_destroy)
        self.renderer = AbstractRenderer.impl(emu)
        self.renderer.init()
        
        # Gtk Overlay Widgets Setup #
        self.overlay_container = Gtk.Overlay()
        self.add(self.overlay_container)
        self.game_area = Gtk.DrawingArea()
        self.game_area.set_size_request(self.total_width * self.scale, self.total_height * self.scale)
        self.game_area.connect("draw", self.on_draw_main)
        self.overlay_container.add(self.game_area)
        self.overlays: dict[str, AsyncOverlay] = {}
        for overlay_cls in overlays:
            self.add_async_overlay(overlay_cls, emu, device=device, refresh_rate=60)
        
    def detach_container(self):
        self.remove(self.overlay_container)
        return self.overlay_container

    # --- NEW METHOD ---
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

    def add_async_overlay(self, overlay_cls: type[AsyncOverlay], emu, device, refresh_rate):
        name = overlay_cls.__name__
        overlay = overlay_cls(emu, device=device, refresh_rate=refresh_rate)
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
            ctx.translate(self.total_width - self.width, self.total_height - self.height) 
            self.renderer.screen(target_width, target_height, ctx, 1)
            ctx.restore()
        
        
        return True
        
class RealWindow(Gtk.Window, EmulatorWindowImpl):
    def __init__(self, *args, **kwargs):
        Gtk.Window.__init__(self)
        self.setup_emulator(*args, **kwargs)
    
        
        
class VirtualWindow(Gtk.OffscreenWindow, EmulatorWindowImpl):
    def __init__(self, *args, **kwargs):
        assert 'id' in kwargs, 'Virtual window must have an id'
        Gtk.OffscreenWindow.__init__(self)
        self.setup_emulator(*args, **kwargs)
        self.id = kwargs['id']
        
    def on_draw_main(self, widget: Gtk.DrawingArea, ctx: cairo.Context, *args):
        stop = SharedMemory(name="stop", size=1)
        assert stop.buf is not None
        if stop.buf[0] == 1:
            Gtk.main_quit()
            return False
        
        original_scale = self.scale
        surface_ctx = cairo.Context(self.shm_surface)
        try:
            self.scale = 1
            result =  super().on_draw_main(widget, surface_ctx, *args)
        finally:
            self.scale = original_scale
        
        ctx.set_source_surface(self.shm_surface, 0, 0)
        ctx.paint()
        return True
        
class VirtualView(Gtk.Overlay, EmulatorContainerImpl):
    def __init__(self, 
        width: int,
        height: int,
        orientation: Literal['horizontal', 'vertical'],
        scale: int,
        id: int | None
    ):
        Gtk.Overlay.__init__(self)
        self.setup_container(width, height, orientation, scale, id)
        self.id = id
        self.game_area = Gtk.DrawingArea()
        self.game_area.set_size_request(self.total_width * self.scale, self.total_height * self.scale)
        self.game_area.connect("draw", self.on_draw_main)
        
        self.add(self.game_area)
        self.show_all()
        
    def on_draw_main(self, widget: Gtk.DrawingArea, ctx: cairo.Context, *args):
        
        
        #ctx.scale(self.scale, self.scale)
        ctx.set_source_surface(self.shm_surface, 0, 0)
        ctx.paint()
        self.game_area.queue_draw()
        return True
        
        
        
    

@overload
def EmulatorWindow(
    emu: MarioKart, overlays: list[type[AsyncOverlay]], width: int, height: int, orientation: Literal['horizontal', 'vertical'], scale: int, 
    virtual_screen: Literal[True],
    id: int = 0, device=None
) -> VirtualWindow: ...

@overload
def EmulatorWindow(
    emu: MarioKart, overlays: list[type[AsyncOverlay]], width: int, height: int, orientation: Literal['horizontal', 'vertical'], scale: int, 
    virtual_screen: Literal[False] = False,
    id: int = 0, device=None
) -> RealWindow: ...
        
def EmulatorWindow(
    emu: MarioKart, 
    overlays: list[type[AsyncOverlay]], 
    width: int,
    height: int,
    orientation: Literal['horizontal', 'vertical'],
    scale: int,
    virtual_screen: bool = False,
    id: int=0,
    device=None
):
    if virtual_screen:
        return VirtualWindow(emu, overlays, width, height, orientation, scale, device=device, id=id)
    else:
        return RealWindow(emu, overlays, width, height, orientation, scale, device=device, id=None)
        
class MultiEmulatorWindow(Gtk.Window):
    def __init__(self, 
        width: int,
        height: int,
        orientation: Literal['horizontal', 'vertical'],
        scale: int,
        n_windows: int
    ):
        super().__init__()
        self.set_title(f"Multi-View ({n_windows})")
        
        self.grid = Gtk.Grid()
        #self.grid.set_column_homogeneous(True)
        #self.grid.set_row_homogeneous(True)
        self.set_default_size(SCREEN_WIDTH * 3, SCREEN_HEIGHT * 3)
        self.add(self.grid)

        # --- CALCULATION LOGIC ---
        # 1. Calculate columns based on square root
        #    ceil(sqrt(5)) = 3 columns
        #    ceil(sqrt(16)) = 4 columns
        n_cols = math.ceil(math.sqrt(n_windows))
        
        # 2. Calculate rows based on columns
        #    If we have 5 windows and 3 cols, we need ceil(5/3) = 2 rows.
        if n_cols > 0:
            n_rows = math.ceil(n_windows / n_cols)
        else:
            n_rows = 0 # Handle 0 windows edge case

        # --- PLACEMENT LOGIC ---
        for i in range(n_windows):
            # Calculate x (col) and y (row) index
            col = i % n_cols
            row = i // n_cols
            
            sub_window = VirtualView(width, height, orientation, scale, id=i)
            
            # Attach to grid: child, left, top, width, height
            self.grid.attach(sub_window, col, row, 1, 1)
            print("test")

        self.connect("destroy", self.on_destroy)
        self.show_all()
        
    def on_destroy(self, widget: Gtk.Widget):
        for child in self.grid.get_children():
            child.destroy()
            
        Gtk.main_quit()

    
                
def main_mp():
    from src.main import main, MainKwargs
    from pathlib import Path
    kwargs: MainKwargs = {
        "rom_path": Path("private/mariokart_ds.nds"),
        "movie": None,
        "record": None,
        "record_data": None,
        "fps": None,
        "sram": None,
        "force_overlay": False,
        "savestate": None,
        "id": 0
    }
    
    
    stop = SharedMemory(name="stop", create=True, size=1)
    assert stop.buf is not None
    stop.buf[0] = 0
    
    processes = []
    shm_names = []
    for i in range(4):
        shm_name = f"display_{i}"
        try:
            SharedMemory(
                name=shm_name,
                create=True,
                size=SCREEN_WIDTH * SCREEN_HEIGHT * 8
            )
        except:
            SharedMemory(
                name=shm_name,
                create=False,
                size=SCREEN_WIDTH * SCREEN_HEIGHT * 8
            )
        p_kwargs = kwargs.copy()
        p_kwargs['id'] = i
        process = Process(
            target=main,
            kwargs=p_kwargs,
            daemon=True
        )
        processes.append(process)
        shm_names.append(shm_name)
        
    
    # Start processes
    for p in processes:
        p.start()
        
    win = MultiEmulatorWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "horizontal",1, 1) # Should result in a 3x2 grid (3 cols, 2 rows)
    win.show_all()
    Gtk.main()
    
    stop.buf[0] = 1
    
    for p in processes:
        p.join()
        
    print("All child processes cleaned up.")

    # Join processes
    for p in processes:
        p.join()
        
    for n in shm_names:
        shm = SharedMemory(name=n)
        shm.close()
        shm.unlink()
        
    stop.close()
    stop.unlink()
    
if __name__ == "__main__":
    main_mp()
    
