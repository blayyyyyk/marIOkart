from abc import abstractmethod
from functools import reduce
from typing import cast

import cairo
import gymnasium as gym
import matplotlib.cm as cm
import matplotlib.colors as plt_colors
import numpy as np
import torch
from desmume.emulator_mkds import MarioKart
from desmume.vector import generate_plane_vectors
from gym_mkds.wrappers.sweeping_ray import (
    find_current_boundary_lines,
    get_standing_triangle_id,
)
from PIL import Image, ImageDraw, ImageFont

from mariokart_ml.utils.collision import compute_collision_dists

KEY_LABELS = ['X', 'Y', 'L', 'R', '↓', '↑', '←', '→', '8', '9', 'B', 'A']
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


class ControllerDisplay(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, n_physical_keys: int = 12):
        super(ControllerDisplay, self).__init__(env)

        self.n_physical_keys = n_physical_keys
        self.input_mask = np.zeros((n_physical_keys,), dtype=np.bool)


    def _is_pressed(self, key: str) -> bool:
        """
        Your ambiguous function to check if a key is pressed.
        Replace this logic with whatever state tracks your current inputs!
        """
        # Example: return key in self.unwrapped.current_pressed_keys
        return False

    def observation(self, observation):
        keymask = observation["keymask"]
        binary_string = bin(keymask.tolist()[0])[2:]
        padded_binary = binary_string.zfill(self.n_physical_keys)
        self.input_mask = np.array([int(b) == 1 for b in padded_binary])
        return observation

    def render(self):
        frame = super().render()
        if self.render_mode == "rgb_array" and frame is not None:
            assert isinstance(frame, np.ndarray) and not isinstance(frame, list)

            H = 40
            W = frame.shape[1]
            C = frame.shape[2]
            N = self.input_mask.shape[0]

            dashboard = np.zeros((H, W, C), dtype=frame.dtype)
            button_width = W // N
            pad = W % N

            active_area = dashboard[:, :W - pad, :].reshape(H, N, button_width, C)
            active_area[:, self.input_mask, :, :] = 255

            pil_dash = Image.fromarray(dashboard)
            draw = ImageDraw.Draw(pil_dash)

            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except IOError:
                try:
                    font = ImageFont.truetype("/Users/blakemoody/Library/Fonts/JetBrainsMonoNerdFont-Regular.ttf", 20)
                except IOError:
                    # font fallback
                    font = ImageFont.load_default()

            for i in range(N):
                label = KEY_LABELS[i] if i < len(KEY_LABELS) else "?"

                bbox = draw.textbbox((0, 0), label, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]

                text_x = (i * button_width) + (button_width // 2) - (text_w // 2)
                text_y = (H // 2) - (text_h // 2) - 2

                is_pressed = self.input_mask[i]
                fontColor = (0, 0, 0) if is_pressed else (255, 255, 255)

                draw.text((text_x, text_y), label, font=font, fill=fontColor)

            dashboard = np.array(pil_dash)
            alpha = 0.5
            frame[:H, :, :] = (frame[:H, :, :] * (1 - alpha)) + (dashboard * alpha)

            return frame

        return frame


class RewardDisplayWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.info = {}

    def step(self, action):
        # Unpack dynamically to support both older Gym (4-tuple) and newer Gymnasium (5-tuple)
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.info = info

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        # Reset the score at the start of a new episode
        self.cumulative_reward = 0.0
        return self.env.reset(**kwargs)

    def render(self):
        frame = super().render()

        if getattr(self, "render_mode", None) == "rgb_array" and frame is not None:
            assert isinstance(frame, np.ndarray) and not isinstance(frame, list)

            # Convert numpy array to PIL Image
            # Ensure it's RGBA so we can easily composite a semi-transparent background
            pil_frame = Image.fromarray(frame).convert("RGBA")

            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except IOError:
                try:
                    font = ImageFont.truetype("/Users/blakemoody/Library/Fonts/JetBrainsMonoNerdFont-Regular.ttf", 12)
                except IOError:
                    font = ImageFont.load_default()


            text = f""
            for k, v in self.info.items():
                if isinstance(v, float):
                    text += f"{k}: {v}\n"

            # Create a separate transparent overlay for the text and its background
            overlay = Image.new('RGBA', pil_frame.size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(overlay)

            # Calculate text bounding box
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            # Calculate top-right position
            padding = 10
            img_w, _ = pil_frame.size
            x = img_w - text_w - padding
            y = padding

            # Draw a semi-transparent black rectangle behind the text for readability
            rect_pad = 4
            draw.rectangle(
                [x - rect_pad, y - rect_pad, x + text_w + rect_pad, y + text_h + rect_pad],
                fill=(0, 0, 0, 150) # 150/255 opacity
            )

            # Draw the white text
            draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))

            # Composite the overlay onto the original frame and convert back to RGB array
            composite = Image.alpha_composite(pil_frame, overlay).convert("RGB")
            return np.array(composite)

        return frame


class CairoWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super(CairoWrapper, self).__init__(env)
        self.depth_mask = True
        self.prev_observation = None

    def __call__(self, env: gym.Env) -> gym.Env:
        return self.__class__(env)

    def _project(self, *pts: np.ndarray, colors: np.ndarray, depth_mask=True) -> tuple[np.ndarray, ...]:
        B, C = pts[0].shape
        pts_cat = np.concat(pts, axis=0)
        emu: MarioKart = cast(MarioKart, self.get_wrapper_attr('emu'))
        proj = emu.memory.project_to_screen(pts_cat)
        proj_mask = proj["mask"].reshape(len(pts), B).all(axis=0)
        proj_pts = proj["screen"].reshape(len(pts), B, C)
        proj_pts = proj_pts[:, proj_mask, :] if depth_mask else proj_pts
        proj_colors = colors[proj_mask] if depth_mask else colors
        chunks = np.split(proj_pts, len(pts), axis=0)
        return tuple([x.squeeze(0) for x in chunks]) + (proj_colors,)

    @abstractmethod
    def _compute(self) -> tuple[np.ndarray, ...]:
        ...

    def observation(self, observation):
        return observation

    def render(self):
        if self.render_mode == "rgb_array":
            raw_rgb = cast(np.ndarray, super().render())
            emu: MarioKart = self.get_wrapper_attr('emu')
            if not emu.memory.race_ready:
                return raw_rgb

            h, w, _ = raw_rgb.shape
            arr = np.zeros((h, w, 4), dtype=np.uint8)
            arr[:, :, :3] = raw_rgb

            surface = cairo.ImageSurface.create_for_data(arr, cairo.FORMAT_RGB24, w, h)
            ctx = cairo.Context(surface)

            out = self._compute()
            if len(out) == 2:
                pts, colors = out
                pts, colors = self._project(pts, colors=colors, depth_mask=self.depth_mask)
                draw_points(ctx, pts[0], colors, radius_scale=5.0)
            elif len(out) == 3:
                pts1, pts2, colors = out
                pts1, pts2, colors = self._project(pts1, pts2, colors=colors, depth_mask=self.depth_mask)
                draw_lines(ctx, pts1, pts2, colors)
            elif len(out) == 4:
                pts1, pts2, pts3, colors = out
                pts1, pts2, pts3, colors = self._project(pts1, pts2, pts3, colors=colors, depth_mask=self.depth_mask)
                draw_triangles(ctx, pts1, pts2, pts3, colors)
            else:
                raise ValueError("Overlay wrapper draw function must return tuple of size 2, 3, or 4")

            surface.flush()
            return arr[:, :, :3]

class CollisionPrisms(CairoWrapper):
    def _compute(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        emu: MarioKart = self.get_wrapper_attr('emu')
        idx = get_standing_triangle_id(emu)
        faces = emu.memory.kcl.triangular_faces[idx:idx+1]

        v1, v2, v3 = np.unstack(faces, axis=1)
        tri = np.stack([v1, v2, v3], axis=0)

        # material color
        color_map = np.array(COLOR_MAP, dtype=np.float32) / 255
        collision_type = emu.memory.collision_data["prism_attribute"]["collision_type"][idx:idx+1]
        floor_mask = (
            emu.memory.collision_data["prism_attribute"]["is_floor"] == 1
        )
        wall_mask = emu.memory.collision_data["prism_attribute"]["is_wall"] != 1
        collision_type = collision_type
        colors = color_map[collision_type]
        tri = tri
        v1, v2, v3 = np.unstack(tri)
        return v1, v2, v3, colors


class TrackBoundary(CairoWrapper):
    def _compute(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        emu: MarioKart = cast(MarioKart, self.get_wrapper_attr('emu'))
        boundary_lines = find_current_boundary_lines(emu)
        p0, p1 = np.split(boundary_lines, 2, axis=1)
        p0 = p0.reshape(-1, 3) # squeeze
        p1 = p1.reshape(-1, 3) # squeeze

        colors = np.array([0.0, 0.1, 0.9])[None, :].repeat(p0.shape[0], axis=0)
        return p0, p1, colors


class OverlayWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, *overlay_classes: CairoWrapper):
        super(OverlayWrapper, self).__init__(reduce(lambda e, cls: cls(e), overlay_classes, env))
    

class SweepingRayOverlay(CairoWrapper):
    # disable depth mask
    def __init__(self, env: gym.Env, n_rays: int, color_map: str = "viridis"):
        super(SweepingRayOverlay, self).__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Dict)
        self.depth_mask = False
        self.color_map = color_map
        self.n_rays = n_rays

    def _compute(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        emu: MarioKart = self.get_wrapper_attr('emu')
        n_rays: int = self.n_rays

        # driver info
        position = emu.memory.driver_position
        mtx = emu.memory.driver_matrix2 # 3x3 rotation matrix (row-major)

        # ray generation
        ray_origin, ray_direction = generate_plane_vectors(n_rays, 180, mtx, position)
        p1 = ray_origin

        # collect ray intersections
        t = compute_collision_dists(emu, n_rays=n_rays, mode="nearest")
        if t is None:
            t = np.zeros(n_rays)

        p0 = ray_direction * t[:, None] + p1

        # color map
        norm = plt_colors.Normalize(vmin=t.min(), vmax=t.max())
        cmap = cm.get_cmap(self.color_map)
        colors = cmap(norm(t))[:, :3]

        return p0, p1, colors


class CheckpointOverlay(CairoWrapper):
    def _compute(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        emu: MarioKart = self.get_wrapper_attr('emu')
        checkpoint_info = emu.memory.checkpoint_info()
        p0, p1 = np.split(checkpoint_info["next_checkpoint_pos"], 2, axis=0)

        return p0, p1, np.array([0.5, 0.7, 0.0])[None, :]
