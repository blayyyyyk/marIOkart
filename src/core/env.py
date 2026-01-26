from typing import Optional
from core.emulator import MarioKart
import numpy as np
import gymnasium as gym
from src.core.memory import *
from typing import Optional
from multiprocessing.shared_memory import SharedMemory
from threading import Thread
from core.window import SharedEmulatorWindow, on_draw_memoryview
from src.visualization.overlay import AVAILABLE_OVERLAYS
from desmume.frontend.gtk_drawing_area_desmume import AbstractRenderer
from desmume.emulator import SCREEN_HEIGHT, SCREEN_WIDTH, SCREEN_PIXEL_SIZE
from queue import Queue
from src.main import EmulatorWindow, OVERLAYS

DEVICE = "cpu"

class MarioKart_Env(gym.Env):
    def __init__(self, rom_file: str, state_file: str, keymap: dict[int, int] | list[int], history_size: int, render_mode: str):
        self.emu = MarioKart()
        self.emu.open(rom_file)
        self.state_file = state_file
        self.history_size = history_size
        self.emu.savestate.load_file(state_file)
        self.keymap = keymap

        self._obs_history = torch.randn((history_size, 3))
        self._action_history = torch.randint(0, len(keymap), (history_size+1, 1))

        self.observation_space = gym.spaces.Dict(
            {
                "wall_distances": gym.spaces.Box(0, 3000, shape=(history_size, 3), dtype=np.float32),
                "actions": gym.spaces.Sequence(
                    gym.spaces.Discrete(9),
                    stack=True
                )
            }
        )

        self.action_space = gym.spaces.Discrete(9)

        self._keymap = keymap

        self._initial_progress: float = self.emu.memory.race_status.driverStatus[0].raceProgress / 0x1000
        self._progress = self._initial_progress
        self._good_reward = 0
        self._bad_reward = 0

    def _get_obs(self):
        fwd = read_forward_distance_obstacle(self.emu, device=DEVICE, interval=(-0.1, 0.1), n_steps=24, sweep_plane="xz")
        left = read_left_distance_obstacle(self.emu, device=DEVICE, interval=(-0.1, 0.1), n_steps=24, sweep_plane="xz")
        right = read_right_distance_obstacle(self.emu, device=DEVICE, interval=(-0.1, 0.1), n_steps=24, sweep_plane="xz")
        new_dist = torch.cat([left, fwd, right], dim=-1)
        new_dist[new_dist > 5000] = 5000
        return {
            "wall_distances": torch.cat([self._obs_history[1:], new_dist[None, :]], dim=0),
            "actions": self._action_history[:-1]
        }
    
    def _get_info(self):
        clock = self.emu.get_ticks()

        return {
            "clock": clock,
            "progress": self._progress - self._initial_progress,
        }
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._obs_history = torch.randn((self.history_size, 3))
        self._action_history = torch.randint(0, len(self.keymap), (self.history_size+1, 1))
        self.emu.savestate.load_file(self.state_file)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):

        self._obs_history = self._get_obs()["wall_distances"]
        self._action_history[:-1] = self._action_history[1:].clone()
        self._action_history[-1] = action

        keymask = self._keymap[action]
        self.emu.input.keypad_update(0)
        self.emu.input.keypad_update(keymask)
        self.emu.cycle()

        observation = self._get_obs()
        info = self._get_info()

        old_progress = self._progress
        new_progress = self.emu.memory.race_status.driverStatus[0].raceProgress / 0x1000 - self._initial_progress
        self._progress = new_progress
        
        progress_delta: float = new_progress - old_progress
        if progress_delta > 0:
            self._good_reward += progress_delta
        else:
            self._bad_reward += abs(progress_delta)

        

        termination_threshold = abs(self._initial_progress) # this is a hyperparam
        terminated = (self._good_reward - self._bad_reward) + termination_threshold < 0
        truncated = False

        reward = -1.0 if terminated else progress_delta

        return observation, reward, terminated, truncated, info