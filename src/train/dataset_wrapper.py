import shutil
import gymnasium as gym
from gymnasium.wrappers.utils import RunningMeanStd
import cairo, torch, os
import numpy as np
from typing import Any, cast, Callable, TypedDict, Optional
from gymnasium.wrappers import NormalizeObservation
import json
import pynput
from desmume.controls import Keys, keymask
from src.config import *


class DatasetWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, out_path: str, max_steps: int = 100000):
        super(DatasetWrapper, self).__init__(env)
        self.out_path = out_path
        self.max_steps = max_steps
        self.current_step = 0
        self.mmaps = {}
        self.mdata = {}
        self.progress = 0.0
        if isinstance(env.observation_space, gym.spaces.Dict):
            spaces = env.observation_space.spaces.items()
        else:
            # not sure if single observation case works, I just did this to get pylance off my back
            spaces = {"default": env.observation_space}.items()

        self.obs_rms = {
            key: RunningMeanStd(shape=space.shape)
            for key, space in spaces
            if space.shape is not None and space.shape[0] > 1
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
            
        # Skip if movie was finished
        if "movie_playing" in info:
            if not info["movie_playing"]:
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

                if key not in self.obs_rms:
                    continue
                batched_value = np.expand_dims(value, axis=0)
                self.obs_rms[key].update(batched_value)

            self.current_step += 1

            # Periodically flush to disk to prevent data loss on crash
            if self.current_step % 1000 == 0:
                for mmap in self.mmaps.values():
                    mmap.flush()

        self.progress = info['race_progress']
        return obs, reward, terminated, truncated, info

    def render(self):
        return super().render()

    def close(self):
        # Ensure all data is written before closing
        for mmap in self.mmaps.values():
            mmap.flush()

        for key in self.mdata.keys():
            self.mdata[key]["shape"] = [self.current_step, *self.mdata[key]["shape"]]

            if key not in self.obs_rms:
                continue
            self.mdata[key]["mean"] = self.obs_rms[key].mean.tolist()
            self.mdata[key]["std"] = np.sqrt(self.obs_rms[key].var).tolist()

        with open(os.path.join(self.out_path, "mdata.json"), "w") as f:
            json.dump(self.mdata, f)

        super().close()
