from gym_mkds.wrappers.window import VecEnvWindow
from abc import ABC
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv, VecEnvWrapper
from typing import Any, TypeVar, Union, cast, Optional

import cairo
import gi
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
import numpy as np
from desmume.emulator import SCREEN_HEIGHT, SCREEN_HEIGHT_BOTH, SCREEN_WIDTH

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gdk, GLib, Gtk # type: ignore


def tile_images(images: np.ndarray) -> np.ndarray:
    """Reduces the image dimensions by tiling the images into a single image."""
    N, H, W, C = images.shape
    cols = int(np.ceil(np.sqrt(N)))
    rows = int(np.ceil(N / cols))
    # add blank tiles (if necessary)
    N_blank = cols * rows - N
    if N_blank > 0:
        blanks = np.zeros((N_blank, H, W, C), dtype=np.uint8)
        images = np.concatenate([images, blanks])

    out = (
        images.reshape(rows, cols, H, W, C)
        .swapaxes(1, 2)
        .reshape(rows * H, cols * W, C)
    )
    return out

ENV_T = TypeVar('ENV_T', bound=Union[VecEnv, gym.Env])

class WindowBase[ENV_T](Gtk.Window):
    def __init__(self, env: ENV_T, scale: float = 1.0):
        super(WindowBase, self).__init__()
        self.set_title("MarioKart Async Training")
        self.width = int(round(SCREEN_WIDTH * scale))
        self.height = int(round(SCREEN_HEIGHT_BOTH * scale))
        self.set_default_size(self.width, self.height)
        self.size_set = None
        self.env = env
        self.is_alive = True
        self.scale = scale

        self.drawing_area = Gtk.DrawingArea()
        self.drawing_area.connect("draw", self.on_draw)
        self.add(self.drawing_area)

        self.connect("destroy", self._on_destroy_main)
        # We don't set a default size yet, we'll let Cairo handle it
        self.show_all()

    def _on_destroy_main(self, widget: Gtk.Widget):
        self.on_destroy()

    def on_destroy(self):
        self.is_alive = False
        self.destroy()
        Gtk.main_quit()

    def update(self):
        # Queue a redraw for the main GTK window
        self.drawing_area.queue_draw()

        # Process GTK events
        while Gtk.events_pending():
            Gtk.main_iteration()

    def on_draw(self, widget: Gtk.Widget, ctx: cairo.Context):
        ...

class Window(WindowBase[gym.Env]):
    def __init__(self, env: gym.Env, scale: float = 1.0):
        super().__init__(env, scale)

    def on_draw(self, widget: Gtk.Widget, ctx: cairo.Context) -> bool:
        arr = cast(np.ndarray, self.env.render())
        if not isinstance(arr, np.ndarray):
            return True

        height, width, _ = arr.shape
        self.resize(width * self.scale, height * self.scale)
        new_arr = np.ones((height, width, 4), dtype=np.uint8)
        new_arr[:, :, :3] = arr
        top = np.ascontiguousarray(new_arr)

        upper_image = cairo.ImageSurface.create_for_data(
            top.data, cairo.FORMAT_RGB24, width, height
        )

        ctx.scale(self.scale, self.scale)
        ctx.set_source_surface(upper_image, 0, 0)
        ctx.get_source().set_filter(cairo.FILTER_NEAREST)
        ctx.paint()

        return True


class VecWindow(WindowBase[VecEnv]):
    def __init__(self, vec_env: VecEnv, scale: float = 1.0, colors: list[tuple[float, float, float]] | None = None):
        super(VecWindow, self).__init__(vec_env, scale=scale)
        # Default colors if none provided (R, G, B normalized 0-1)
        self.colors = colors or [
            (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
            (1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0)
        ]
        self.frame_width = 4 # Border thickness in pixels

    def on_draw(self, widget: Gtk.Widget, ctx: cairo.Context) -> bool:
        if isinstance(self.env, gym.vector.AsyncVectorEnv):
            images = self.env.render()
        else:
            images = self.env.get_images()

        if not isinstance(images, (np.ndarray, tuple, list)):
            return True
        if isinstance(images, (tuple, list)):
            images = np.stack(images)

        if images.ndim == 3:
            images = images[None, ...]

        N, H, W, C = images.shape
        images = images[:, :H//2, :, :]
        H = images.shape[1]
        cols = int(np.ceil(np.sqrt(N)))
        rows = int(np.ceil(N / cols))

        # Update window size once based on the grid + frame spacing
        total_w = cols * (W + self.frame_width * 2)
        total_h = rows * (H + self.frame_width * 2)
        if not self.size_set:
            self.resize(int(total_w * self.scale), int(total_h * self.scale))
            self.size_set = True

        ctx.scale(self.scale, self.scale)

        for i in range(N):
            col = i % cols
            row = i // cols

            x = col * (W + self.frame_width * 2)
            y = row * (H + self.frame_width * 2)

            color = self.colors[i % len(self.colors)]
            ctx.set_source_rgb(*color)
            ctx.rectangle(x, y, W + self.frame_width * 2, H + self.frame_width * 2)
            ctx.fill()

            img = images[i]
            rgba_buffer = np.zeros((H, W, 4), dtype=np.uint8)
            rgba_buffer[:, :, :3] = img
            stride = W * 4
            surface = cairo.ImageSurface.create_for_data(
                np.ascontiguousarray(rgba_buffer), cairo.FORMAT_RGB24, W, H, stride
            )

            ctx.set_source_surface(surface, x + self.frame_width, y + self.frame_width)
            ctx.get_source().set_filter(cairo.FILTER_NEAREST)
            ctx.paint()

            label_text = f"Agent {i}"
            ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
            ctx.set_font_size(12)

            extents = ctx.text_extents(label_text)
            text_x = x + (W + self.frame_width) - extents.width - 5
            text_y = y + extents.height + 5

            ctx.set_source_rgba(0, 0, 0, 0.6)
            ctx.rectangle(text_x - 2, text_y - extents.height - 2, extents.width + 4, extents.height + 4)
            ctx.fill()

            ctx.set_source_rgb(*color)
            ctx.move_to(text_x, text_y)
            ctx.show_text(label_text)

        return True


class WindowWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, scale: float = 1.0):
        super().__init__(env)
        self.window = None
        self.scale = scale
        
    def reset(self, *args, seed=None, options=None):
        obs, info = super().reset()
        self.window = Window(self.env, scale=self.scale)
        return obs, info
        
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        
        if self.window is not None:
            self.window.update()
        
        return obs, reward, terminated, truncated, info
        


class VecWindowWrapper(gym.vector.VectorWrapper):
    env: VecEnv
    window: Optional[VecWindow]
    
    def __init__(self, env: AsyncVectorEnv, scale: float = 1.0, colors: list[tuple[float, float, float]] | None = None):
        super().__init__(env)
        self.window = None
        self.scale = scale
        self.colors = colors
        
    def close(self, **kwargs):
        if self.window is None: return
        self.window.destroy()
        super().close()
        
    def reset(self, *args, seed=None, options=None):
        obs, info = super().reset()
        self.window = VecWindow(self.env, scale=self.scale, colors=self.colors) if self.window is None else self.window
        return obs, info
        
    def step(self, actions):
        obs, reward, terminated, truncated, info = super().step(actions)
        
        if self.window is not None:
            self.window.update()
        
        return obs, reward, terminated, truncated, info
        
class VecWindowWrapperSB3(VecEnvWrapper):
    window: Optional[VecWindow]
    
    def __init__(self, venv: SubprocVecEnv, scale: float, colors: list[tuple[float, float, float]] | None = None):
        # SB3 wrappers take 'venv' instead of 'env'
        super().__init__(venv)
        self.window = None
        self.scale = scale
        self.colors = colors
        

    def reset(self):
        obs = self.venv.reset()
        self.window = VecWindow(self.venv, scale=self.scale, colors=self.colors) if self.window is None else self.window
        return obs

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        
        if self.window is not None:
            self.window.update()
        
        return obs, rewards, dones, infos
        
        
