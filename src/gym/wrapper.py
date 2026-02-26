import gymnasium as gym
from gymnasium.wrappers.utils import RunningMeanStd
import cairo, torch, os
import numpy as np
from typing import Any, cast, Callable, TypedDict
from gymnasium.wrappers import NormalizeObservation
from src.core.emulator import MarioKart
from gym.env import MarioKartEnv
import json

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


def draw_points(
    ctx: cairo.Context,
    pts: np.ndarray,
    colors: np.ndarray,
    radius_scale: float | np.ndarray,
):
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


def draw_lines(
    ctx: cairo.Context,
    pts1: np.ndarray,
    pts2: np.ndarray,
    colors: np.ndarray,
    stroke_width_scale=1.0,
):
    if pts1.ndim == 1:
        pts1 = pts1[None, :]

    if pts2.ndim == 1:
        pts2 = pts2[None, :]

    assert (
        pts2.shape[0] == pts1.shape[0]
    ), "All point arrays must have the same batch size"

    if colors.ndim == 1:
        colors = colors[None, :]

    assert colors.shape[0] == 1 or colors.shape[0] == pts1.shape[0]
    if colors.shape[0] == 1:
        colors = colors.repeat(pts1.shape[0], axis=0)

    for p1, p2, (r, g, b) in zip(pts1, pts2, colors):
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
    colors: np.ndarray,
):
    n = pts1.shape[0]
    assert (
        pts2.shape[0] == n and pts3.shape[0] == n
    ), "All point arrays must have the same batch size"

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


class OverlayOutput(TypedDict):
    triangles: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None
    lines: tuple[np.ndarray, np.ndarray, np.ndarray] | None
    points: tuple[np.ndarray, np.ndarray] | None


class OverlayWrapper(gym.Wrapper):
    def __init__(self, env, func: Callable[[Any], OverlayOutput]):
        super(OverlayWrapper, self).__init__(env)
        self.env = env
        self.func = func

    def render(self):
        if self.render_mode == "rgb_array":
            raw_rgb = cast(np.ndarray, super().render())
            if not self.env.emu.memory.race_ready:
                return raw_rgb

            h, w, _ = raw_rgb.shape
            arr = np.zeros((h, w, 4), dtype=np.uint8)
            arr[:, :, :3] = raw_rgb

            surface = cairo.ImageSurface.create_for_data(arr, cairo.FORMAT_RGB24, w, h)
            ctx = cairo.Context(surface)

            out = self.func(self.env)
            if out["points"] is not None:
                pts, colors = out["points"]
                draw_points(ctx, pts, colors, radius_scale=5.0)

            if out["lines"] is not None:
                pts1, pts2, colors = out["lines"]
                draw_lines(ctx, pts1, pts2, colors)

            if out["triangles"] is not None:
                pts1, pts2, pts3, colors = out["triangles"]
                draw_triangles(ctx, pts1, pts2, pts3, colors)

            surface.flush()
            return arr[:, :, :3]


def collision_overlay(env) -> OverlayOutput:
    v1 = env.emu.memory.collision_data["v1"].to(env.emu.device)
    v2 = env.emu.memory.collision_data["v2"].to(env.emu.device)
    v3 = env.emu.memory.collision_data["v3"].to(env.emu.device)
    group = torch.cat([v1, v2, v3], dim=0)
    v_proj = env.emu.memory.project_to_screen(group, normalize_depth=True)

    # z filter
    m1, m2, m3 = v_proj["mask"].chunk(3, dim=0)
    z_mask = m1 & m2 & m3
    tri = v_proj["screen"].view(3, v1.shape[0], -1)
    tri = tri[:, z_mask]

    if tri.shape[0] == 0:
        return {"points": None, "lines": None, "triangles": None}

    # material color
    color_map = torch.tensor(COLOR_MAP, dtype=torch.uint8, device=env.emu.device)
    collision_type = env.emu.memory.collision_data["prism_attribute"]["collision_type"]
    collision_type = collision_type[z_mask]
    floor_mask = (
        env.emu.memory.collision_data["prism_attribute"]["is_floor"][z_mask] == 1
    )
    wall_mask = env.emu.memory.collision_data["prism_attribute"]["is_wall"][z_mask] != 1
    collision_type = collision_type[floor_mask & wall_mask]
    color_ids = torch.tensor(collision_type, dtype=torch.int32, device=env.emu.device)
    colors = color_map[color_ids]
    tri = tri[:, floor_mask & wall_mask]

    colors = colors.detach().cpu().numpy()
    tri = tri.detach().cpu().numpy()
    v1, v2, v3 = np.unstack(tri)

    return {"points": None, "lines": None, "triangles": (v1, v2, v3, colors)}


def sensor_overlay(env) -> OverlayOutput:
    P = env.emu.memory.driver.position.to(env.emu.device)
    P = P.unsqueeze(0)
    max_dist = env.emu.max_dist
    n_rays = env.emu.n_rays
    info = env.emu.memory.obstacle_info(
        n_rays=n_rays, max_dist=max_dist, device=env.emu.device
    )
    R, D = info["position"], info["distance"]

    R_s = env.emu.memory.project_to_screen(R, normalize_depth=True)["screen"]
    P_s = env.emu.memory.project_to_screen(P, normalize_depth=True)["screen"]
    P_s = P_s.expand_as(R_s)
    P = P.expand_as(R)
    colors = torch.tensor([1.0, 0.0, 0.0]).to(env.emu.device)
    colors = colors.expand_as(P)
    weight = (D - D.mean()) / D.std()
    weight = weight.clamp(0, 1.0)
    colors = colors.clone()
    colors[:, 0] -= weight
    colors[:, 1] += weight

    lines = (
        R_s.detach().cpu().numpy(),
        P_s.detach().cpu().numpy(),
        colors.detach().cpu().numpy(),
    )

    return {"points": None, "lines": lines, "triangles": None}


def compose_overlays(env, funcs: list[Callable[[Any], OverlayOutput]]) -> OverlayOutput:
    """Executes multiple overlay functions and merges their outputs into one."""

    # Containers for the merged data
    merged_points = {"pts": [], "colors": []}
    merged_lines = {"pts1": [], "pts2": [], "colors": []}
    merged_triangles = {"pts1": [], "pts2": [], "pts3": [], "colors": []}

    for func in funcs:
        out = func(env)

        # Merge Points
        if out.get("points") is not None:
            assert out["points"] is not None
            pts, colors = out["points"]
            merged_points["pts"].append(pts)
            merged_points["colors"].append(colors)

        # Merge Lines
        if out.get("lines") is not None:
            assert out["lines"] is not None
            pts1, pts2, colors = out["lines"]
            merged_lines["pts1"].append(pts1)
            merged_lines["pts2"].append(pts2)
            merged_lines["colors"].append(colors)

        # Merge Triangles
        if out.get("triangles") is not None:
            assert out["triangles"] is not None
            pts1, pts2, pts3, colors = out["triangles"]
            merged_triangles["pts1"].append(pts1)
            merged_triangles["pts2"].append(pts2)
            merged_triangles["pts3"].append(pts3)
            merged_triangles["colors"].append(colors)

    # Final helper to concatenate list of arrays if they exist
    def concat_or_none(arrays):
        return np.concatenate(arrays, axis=0) if arrays else None

    return cast(
        OverlayOutput,
        {
            "points": (
                (
                    concat_or_none(merged_points["pts"]),
                    concat_or_none(merged_points["colors"]),
                )
                if merged_points["pts"]
                else None
            ),
            "lines": (
                (
                    concat_or_none(merged_lines["pts1"]),
                    concat_or_none(merged_lines["pts2"]),
                    concat_or_none(merged_lines["colors"]),
                )
                if merged_lines["pts1"]
                else None
            ),
            "triangles": (
                (
                    concat_or_none(merged_triangles["pts1"]),
                    concat_or_none(merged_triangles["pts2"]),
                    concat_or_none(merged_triangles["pts3"]),
                    concat_or_none(merged_triangles["colors"]),
                )
                if merged_triangles["pts1"]
                else None
            ),
        },
    )


class DatasetWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, out_path: str, max_steps: int = 100000):
        super(DatasetWrapper, self).__init__(env)
        self.out_path = out_path
        self.max_steps = max_steps
        self.current_step = 0
        self.mmaps = {}
        self.mdata = {}
        self.obs_rms = {
            key: RunningMeanStd(shape=space.shape)
            for key, space in env.observation_space.spaces.items()
        }
        
        

        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)

    def _init_mmaps(self, obs):
        """Initialize memmap files based on the first observation's structure."""
        for key, value in obs.items():
            file_path = os.path.join(self.out_path, f"{key}.dat")

            # Determine shape: (Max Steps, *Observation Shape)
            # e.g., (100000, 20) for your wall_distances
            shape = (self.max_steps, *value.shape)

            # Create the memmap file
            self.mmaps[key] = np.memmap(
                file_path, dtype=value.dtype, mode="w+", shape=shape
            )

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Initialize files on the first step
        if not self.mmaps:
            self._init_mmaps(obs)

        # Skip menu frames if they exist
        skip_write = False
        if "race_started" in info:
            if not info["race_started"]:
                skip_write = True

        # Write each observation component to its respective .dat file
        if self.current_step < self.max_steps and not skip_write:
            for key, value in obs.items():
                self.mmaps[key][self.current_step] = value

                if key not in self.mdata:
                    self.mdata[key] = {
                        "shape": value.shape,
                        "dtype": value.dtype.name,
                    }
                
                if key not in self.obs_rms: continue
                batched_value = np.expand_dims(value, axis=0)
                self.obs_rms[key].update(batched_value)

            self.current_step += 1

            
            # Periodically flush to disk to prevent data loss on crash
            if self.current_step % 1000 == 0:
                for mmap in self.mmaps.values():
                    mmap.flush()

        return obs, reward, terminated, truncated, info

    def render(self):
        return super().render()

    def close(self):
        # Ensure all data is written before closing
        for mmap in self.mmaps.values():
            mmap.flush()

        for key in self.mdata.keys():
            self.mdata[key]["shape"] = [self.current_step, *self.mdata[key]["shape"]]
            
            if key not in self.obs_rms: continue
            self.mdata[key]["mean"] = self.obs_rms[key].mean.tolist()
            self.mdata[key]["std"] = np.sqrt(self.obs_rms[key].var).tolist()

        with open(os.path.join(self.out_path, "mdata.json"), "w") as f:
            json.dump(self.mdata, f)

        super().close()
