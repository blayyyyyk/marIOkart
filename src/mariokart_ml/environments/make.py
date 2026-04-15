from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import Autoreset
from stable_baselines3.common.vec_env import SubprocVecEnv

from mariokart_ml.wrappers.overlay_wrapper import OverlayWrapper

from ..config import N_KEYS, RAY_COUNT, SAVE_STATE_SAMPLE_COUNT, SPARSE_KEYMAP
from ..wrappers import *


def _make(
    env_name: str,
    mode: Literal['play', 'menu', 'movie', 'train'],
    autoreset: bool = False,
    movie: Optional[Path] = None,
) -> gym.Env[dict[str, Any], int]:
    env = gym.make(env_name)
    if mode == 'movie':
        env = MovieWrapper(env, str(movie), max_steps="race_end")
    elif mode == 'menu':
        env = MovieWrapper(env, str(movie), max_steps="race_start")
    elif mode == "train" and movie is not None:
        env = MovieWrapper(env, str(movie), max_steps="lap_end")
    elif mode == 'play': pass
    else:
        raise ValueError(f"Invalid debug mode provided")

    if mode in ('play', 'menu') or (mode == "train" and movie is None):
        env = KeyboardWrapper(env)

    if autoreset:
        env = Autoreset(env)



    if mode == 'train':
        env = SaveStateSampling(env, n_samples=SAVE_STATE_SAMPLE_COUNT)
        env = ControllerRemap(env, keymap=SPARSE_KEYMAP)

    return env

def _add_window(
    env: gym.Env,
    show_keys: bool,
    show_boundary: bool,
    show_rays: bool
):
    # special overlay (requires input monitoring)
    if show_keys:
        env = ControllerObservation(env, n_keys=N_KEYS)
        env = ControllerDisplay(env, n_physical_keys=N_KEYS)

    # generic overlays (draws exclusive shapes)
    overlays = []
    if show_boundary:
        overlays.append(TrackBoundary)

    if show_rays:
        overlays.append(partial(SweepingRayOverlay, n_rays=RAY_COUNT))

    if len(overlays) == 0:
        return env

    return OverlayWrapper(env, *overlays)


class EnvManager:
    def __init__(self, env_name: str, mode: Literal['play', 'menu', 'movie', 'train'], autoreset: bool = False, ):
        self.base_factory = partial(_make, env_name, mode, autoreset)

    def make(self, movies: list[Optional[Path]], env_modifier: Optional[Callable] = None, vec_class: type[SubprocVecEnv] | type[AsyncVectorEnv] = gym.vector.AsyncVectorEnv):
        factory = env_modifier if env_modifier else self.base_factory
        
        if len(movies) > 1:
            part = partial(vec_class, [(lambda m=m: factory(m)) for m in movies])
            if vec_class.__name__ == "SubprocVecEnv":
                vec_env = part(start_method='spawn') # sb3
            elif vec_class.__name__ == "AsyncVectorEnv":
                vec_env = part(context='spawn') # gymnasium
            else:
                raise ValueError(f"Unsupported vector environment class: {vec_class.__name__}")

            return vec_env

        return factory(movies[-1])

    def make_windowed(self, movies: list[Optional[Path]], scale: int = 1, show_keys: bool = False, show_boundary: bool = False, show_rays: bool = False):
        windowed_factory = lambda m: _add_window(self.base_factory(m), show_keys, show_boundary, show_rays)
        env = self.make(movies, env_modifier=windowed_factory)
        return EnvManager.add_window(env, scale)

    @staticmethod
    def add_window(env: gym.Env[dict[str, Any], int] | gym.vector.AsyncVectorEnv, scale: int = 1):
        if isinstance(env, gym.vector.AsyncVectorEnv):
            return VecWindowWrapper(env, scale)
        elif isinstance(env, gym.Env):
            return WindowWrapper(env, scale)
        else:
            raise TypeError("env must be a gym.Env or gym.vector.AsyncVectorEnv")
