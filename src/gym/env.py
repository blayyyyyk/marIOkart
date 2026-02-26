import multiprocessing
from multiprocessing.shared_memory import SharedMemory
from desmume.emulator import SCREEN_HEIGHT, SCREEN_HEIGHT_BOTH, SCREEN_PIXEL_SIZE, SCREEN_WIDTH
from desmume.frontend.gtk_drawing_impl.software import cairo
from gi.repository.Gtk import Overlay
import gymnasium as gym
from typing import Any, Callable, Optional, TypedDict, cast
from functools import partial
from gymnasium.vector import AsyncVectorEnv
from src.mkdslib.mkdslib import FX32_SCALE_FACTOR
from desmume.controls import Keys, keymask
from src.core.emulator import MarioKart
import numpy as np
import torch, os
import pynput
from src.gym.window import EnvWindow, VecEnvWindow

from gymnasium.wrappers import HumanRendering


ID_TO_KEY = [0, 33, 289, 1, 257, 321, 801, 273, 17]
KEY_TO_ID = {k: i for i, k in enumerate(ID_TO_KEY)}


        

class MarioKartEnv(gym.Env):
    def __init__(self, rom_path: str, movie_path: Optional[str] = None, ray_max_dist: int = 3000, ray_count: int = 20, render_mode: str = "rgb_array"):
        super().__init__()
        self.device = torch.device("cpu")
        self.emu = MarioKart()
        self.state = self.emu.memory
        self.emu.open(rom_path)
        if movie_path:
            self.movie_path = movie_path
            
        else:
            self.movie_path = None
            
        self.emu.volume_set(0)
        self.ray_count = ray_count
        self.ray_max_dist = ray_max_dist
        self.frame_count = 0
        self.prev_progress = 0.0
        self.observation_space = gym.spaces.Dict(
            {
                "wall_distances": gym.spaces.Box(0, self.ray_max_dist, shape=(ray_count,), dtype=np.float32),
            }
        )
        self.action_space = gym.spaces.Discrete(2048, dtype=np.uint32)
        self.metadata = {
            "render_modes": ["rgb_array"],
            "render_fps": 60
        }
        self.render_mode = render_mode
        
    def _get_obs(self):
        if self.state.race_ready:
            info = self.state.obstacle_info(self.ray_count, self.ray_max_dist, self.device)
            dist = info['distance'].cpu().detach().numpy()
        else:
            dist = np.zeros((self.ray_count,), dtype=np.float32)
        
        keymask = self.emu.input.keypad_get()
        
        return {
            "wall_distances": dist,
            "keymask": np.array([keymask], dtype=np.uint16)
        }
        
    def _get_info(self):
        if self.state.race_ready:
            progress = self.state.race_status.driverStatus[0].raceProgress 
            progress *= FX32_SCALE_FACTOR
        else:
            progress = 0.0
        
        return {
            "race_progress": progress,
            "race_started": self.state.race_ready
        }
        
    def step(self, action):
        if self.emu.movie.is_playing() or action is None:
            action = self.emu.input.keypad_get()
        else:
            self.emu.input.keypad_update(action)
        
        
        self.emu.cycle()
        
        
        obs = self._get_obs()
        info = self._get_info()
        
        if self.state.race_ready:
            curr_progress = self.state.race_status.driverStatus[0].raceProgress
            curr_progress *= FX32_SCALE_FACTOR
            reward = curr_progress - self.prev_progress
            terminated = curr_progress >= 1.0
            self.prev_progress = curr_progress
        else:
            terminated = False
            reward = 0.0
        
        truncated = False
        
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
        
    def render(self):
        mem = self.emu.display_buffer_as_rgbx()
        top = mem[: SCREEN_PIXEL_SIZE * 4]
        bottom = mem[SCREEN_PIXEL_SIZE * 4 :]
        
        # CORRECTED: Shape must be (Height, Width, Channels) to match C-buffers
        arr_t = np.ndarray(shape=(SCREEN_HEIGHT, SCREEN_WIDTH, 4), dtype=np.uint8, buffer=top)
        arr_b = np.ndarray(shape=(SCREEN_HEIGHT, SCREEN_WIDTH, 4), dtype=np.uint8, buffer=bottom)
        
        # CORRECTED: Stack on axis=0 (Height) so the screens are stacked top-to-bottom
        arr = np.concatenate([arr_t, arr_b], axis=0) 
        
        return arr[:, :, :3]
            
        
    def reset(self, seed, options):
        """Gymnasium requires a reset method to restart the environment."""
        super().reset(seed=seed)
        self.emu.reset()
        if self.movie_path:
            self.emu.movie.play(self.movie_path)
        
        self.frame_count = 0
        self.prev_progress = 0.0
        
        # If your emulator class has a reset function, call it here. 
        # e.g., self.emu.reset()
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info
        
    def close(self):
        self.emu.close()
        super().close()




        
def start_keyboard_listener():
    """Starts a non-blocking keyboard listener in a separate thread."""
    input_state = set()

    def on_press(key):
        try:
            name = key.char.lower() if hasattr(key, "char") else key.name
            input_state.add(name)

        except Exception:
            pass

    def on_release(key):
        try:
            name = key.char.lower() if hasattr(key, "char") else key.name
            input_state.discard(name)
        except Exception:
            pass

    listener = pynput.keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.daemon = True
    listener.start()

    return listener, input_state
        
USER_KEYMAP = {
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
