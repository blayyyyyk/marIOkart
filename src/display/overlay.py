from typing import TypeVar, Generic, TypedDict
from multiprocessing.shared_memory import SharedMemory
import threading

from desmume.emulator import SCREEN_WIDTH, SCREEN_HEIGHT
from src.display._impl import AsyncOverlay, draw_lines, draw_triangles, draw_points
import gi
gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gtk, Gdk, GLib
import cairo, time, numpy as np, torch



COLOR_MAP = [
    # --- DRIVEABLE SURFACES (Grays/Whites) ---
    [128, 128, 128],  # 0:  Road (Standard Gray)
    [200, 230, 255],  # 1:  Slippery Road (Icy Blue-White)
    # --- OFFROAD (Greens/Browns) ---
    [34, 139, 34],  # 2:  Weak Offroad (Forest Green - Grass)
    [139, 69, 19],  # 3:  Offroad (Saddle Brown - Mud/Dirt)
    # --- TECHNICAL (Purples) ---
    [238, 130, 238],  # 4:  Sound Trigger (Violet)
    # --- HEAVY OFFROAD ---
    [80, 50, 20],  # 5:  Heavy Offroad (Dark Brown - Deep Mud)
    # --- SLIPPERY VARIANTS ---
    [175, 238, 238],  # 6:  Slippery Road 2 (Pale Turquoise)
    # --- BOOSTS (Oranges) ---
    [255, 140, 0],  # 7:  Boost Panel (Dark Orange)
    # --- WALLS (Reds) ---
    [255, 0, 0],  # 8:  Wall (Pure Red)
    [255, 105, 180],  # 9:  Invisible Wall (Hot Pink - distinct from normal wall)
    # --- BOUNDARIES (Blacks/Darks) ---
    [0, 0, 0],  # 10: Out of Bounds (Black)
    [25, 25, 112],  # 11: Fall Boundary (Midnight Blue - Abyss)
    # --- JUMPS (Yellows) ---
    [255, 255, 0],  # 12: Jump Pad (Yellow)
    # --- AI/DRIVER LOGIC (Tinted Grays) ---
    [169, 169, 169],  # 13: Road (no drivers) (Dark Gray)
    [139, 0, 0],  # 14: Wall (no drivers) (Dark Red)
    # --- MECHANICS (Metals/Indigos) ---
    [75, 0, 130],  # 15: Cannon Activator (Indigo)
    [205, 92, 92],  # 16: Edge Wall (Indian Red)
    # --- WATER ---
    [0, 0, 255],  # 17: Falls Water (Pure Blue)
    # --- BOOST VARIANT ---
    [255, 69, 0],  # 18: Boost Pad w/ Min Speed (Red-Orange)
    # --- SPECIAL ROADS ---
    [192, 192, 192],  # 19: Loop Road (Silver)
    [255, 215, 0],  # 20: Special Road (Gold - e.g., Rainbow Road segments)
    # --- WALL VARIANT ---
    [128, 0, 0],  # 21: Wall 3 (Maroon)
    # --- RECALC ---
    [0, 255, 0],  # 22: Force Recalc (Lime Green - Debug visual)
]




class CheckpointOverlayI(TypedDict):
    layer_1: np.ndarray
    layer_2: np.ndarray


class CheckpointOverlay(AsyncOverlay[CheckpointOverlayI]):
    def compute(self, state) -> CheckpointOverlayI:
        if state is None:
            return {"layer_1": np.zeros((2, 3)), "layer_2": np.zeros((2, 3))}

        device = self.device
        position = self.memory.driver.position
        position = position.to(device)

        # Checkpoint boundary
        checkpoint = self.memory.checkpoint_info()["next_checkpoint_pos"]
        checkpoint = checkpoint.to(device)
        checkpoint[:, 1] -= self.memory.camera.targetElevation / (1 << 12)

        proj = self.memory.project_to_screen(checkpoint, normalize_depth=True)
        screen_space = proj["screen"]
        depth_mask = proj["mask"]
        screen_space = screen_space[depth_mask]
        screen_space = screen_space.detach().cpu().numpy()

        if screen_space.shape[0] == 2:
            assert isinstance(state, dict)
            state["layer_1"] = screen_space

        # Path to checkpoint
        screen_space = None
        intersect = self.memory.read_facing_point_checkpoint(device=device)
        points = torch.stack([intersect, position], dim=0)
        proj = self.memory.project_to_screen(points, normalize_depth=True)
        screen_space = proj["screen"]
        depth_mask = proj["mask"]
        depth_mask[1] = True
        screen_space = screen_space[depth_mask]
        screen_space = screen_space.detach().cpu().numpy()
        if screen_space.shape[0] < 2:
            return state

        state["layer_2"] = screen_space
        return state

    def draw_overlay(self, ctx, state: CheckpointOverlayI):
        layer_1 = state["layer_1"]
        p1, p2 = layer_1[0], layer_1[1]

        layer_2 = state["layer_2"]
        p3, p4 = layer_2[0], layer_2[1]

        p1 = np.stack([p1, p3], axis=0)
        p2 = np.stack([p2, p4], axis=0)
        colors = np.array([0.0, 1.0, 0.0])
        
        draw_lines(ctx, p1, p2, colors=colors, stroke_width_scale=1.0)


CollisionOverlayI = tuple[np.ndarray, np.ndarray]


class CollisionOverlay(AsyncOverlay[CollisionOverlayI]):
    def compute(self, state):
        if state is None:
            return np.zeros((3, 3, 3)), np.zeros((3, 3))

        v1 = self.memory.collision_data["v1"].to(self.device)
        v2 = self.memory.collision_data["v2"].to(self.device)
        v3 = self.memory.collision_data["v3"].to(self.device)
        group = torch.cat([v1, v2, v3], dim=0)
        v_proj = self.memory.project_to_screen(group, normalize_depth=True)

        # z filter
        m1, m2, m3 = v_proj["mask"].chunk(3, dim=0)
        z_mask = m1 & m2 & m3
        tri = v_proj["screen"].view(3, v1.shape[0], -1)
        tri = tri[:, z_mask]

        if tri.shape[0] == 0:
            return self.rendering_state

        # material color
        color_map = torch.tensor(COLOR_MAP, dtype=torch.uint8, device=self.device)
        collision_type = self.memory.collision_data["prism_attribute"]["collision_type"]
        collision_type = collision_type[z_mask]
        floor_mask = (
            self.memory.collision_data["prism_attribute"]["is_floor"][z_mask] == 1
        )
        wall_mask = (
            self.memory.collision_data["prism_attribute"]["is_wall"][z_mask] != 1
        )
        collision_type = collision_type[floor_mask & wall_mask]
        color_ids = torch.tensor(collision_type, dtype=torch.int32, device=self.device)
        colors = color_map[color_ids]
        # colors = colors[z_mask]
        # colors = torch.tensor([[0, 255, 128]], device=self.device)
        tri = tri[:, floor_mask & wall_mask]

        colors = colors.detach().cpu().numpy()
        tri = tri.detach().cpu().numpy()

        return tri, colors

    def draw_overlay(self, ctx, state):
        if not isinstance(state, tuple):
            return

        tri, colors = state
        v1, v2, v3 = np.unstack(tri)
        draw_triangles(ctx, v1, v2, v3, colors)


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
        R, D = info["position"], info["distance"]

        R_s = self.memory.project_to_screen(R, normalize_depth=True)["screen"]
        P_s = self.memory.project_to_screen(P, normalize_depth=True)["screen"]
        P_s = P_s.expand_as(R_s)
        P = P.expand_as(R)
        colors = torch.tensor([1.0, 0.0, 0.0]).to(self.device)
        colors = colors.expand_as(P)
        weight = (D - D.mean()) / D.std()
        weight = weight.clamp(0, 1.0)
        colors = colors.clone()
        colors[:, 0] -= weight
        colors[:, 1] += weight

        return (
            R_s.detach().cpu().numpy(),
            P_s.detach().cpu().numpy(),
            colors.detach().cpu().numpy(),
        )

    def draw_overlay(self, ctx, state):
        p1, p2, colors = state
        draw_lines(ctx, p2, p1, colors=colors, stroke_width_scale=1.0)


class OrientationOverlay(AsyncOverlay[tuple[np.ndarray, np.ndarray]]):
    def compute(self, state):
        if state is None:
            return np.identity(3), np.zeros((3, 3))

        M = self.memory.driver.mainMtx.to(self.device)[:3, :] * 5.0
        pos: torch.Tensor = self.memory.driver.position.to(self.device)

        P = pos.expand_as(M)
        M = self.memory.project_to_screen(M + P, normalize_depth=True)["screen"]
        P = self.memory.project_to_screen(P, normalize_depth=True)["screen"]
        state = M.detach().cpu().numpy(), P.detach().cpu().numpy()
        return state

    def draw_overlay(self, ctx, state):
        M, P = state
        colors = np.identity(3, dtype=np.float32)
        draw_lines(ctx, M, P, colors)


class DriftOverlay(AsyncOverlay[tuple[np.ndarray, np.ndarray]]):
    def compute(self, state):
        if state is None:
            return np.zeros((1, 3)), np.zeros((1, 3))

        D = self.memory.driver.drivingDirection.unsqueeze(0) * 10.0
        P = self.memory.driver.position.unsqueeze(0)

        D = self.memory.project_to_screen(D + P, normalize_depth=True)["screen"]
        P = self.memory.project_to_screen(P, normalize_depth=True)["screen"]

        return D.detach().cpu().numpy(), P.detach().cpu().numpy()

    def draw_overlay(self, ctx, state):
        D, P = state
        colors = np.array([[0.9, 0.1, 0.9]])
        draw_lines(ctx, D, P, colors)
