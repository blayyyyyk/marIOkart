from mariokart_ml.wrappers.controller_wrapper import ControllerDriftingRemap
from functools import partial, reduce
from pathlib import Path
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    TypeAliasType,
    Union,
    cast,
    overload,
)

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import Autoreset
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv

from mariokart_ml.utils.game_event import LapEndEvent, RaceEndEvent, RaceStartEvent
from mariokart_ml.wrappers.overlay_wrapper import OverlayWrapper
from mariokart_ml.wrappers.window_wrapper import VecWindowWrapperSB3

from ..config import N_KEYS, RAY_COUNT, SAVE_STATE_SAMPLE_COUNT, SPARSE_KEYMAP
from ..wrappers import *


def _make(
    env_name: str,
    mode: Literal['play', 'menu', 'movie', 'train'],
    autoreset: bool = False,
    movie: Optional[Path] = None,
    **wrappers: list[gym.Wrapper]
) -> gym.Env[dict[str, Any], int]:
    env = gym.make(env_name)
    if mode == 'movie':
        env = MovieWrapper(env, str(movie), disable_event=RaceEndEvent())
    elif mode == 'menu':
        env = MovieWrapper(env, str(movie), disable_event=RaceStartEvent())
    elif mode == "train":
        if movie is None:
            env = KeyboardWrapper(env, disable_event=LapEndEvent())
        else:
            env = MovieWrapper(env, str(movie), disable_event=LapEndEvent())
    elif mode == 'play': pass
    else:
        raise ValueError(f"Invalid debug mode provided")

    if mode == 'play' or mode == 'menu':
        env = KeyboardWrapper(env, disable_event=LapEndEvent())

    if autoreset:
        env = Autoreset(env)

    if mode == 'train':
        env = SaveStateSampling(env, n_samples=SAVE_STATE_SAMPLE_COUNT, collect_saves_event=LapEndEvent())
        env = ControllerDriftingRemap(env, enable_event=LapEndEvent())
        
    if wrappers is not None:
        env = reduce(lambda e, cls: cls(e), wrappers, env)

    return env

def _make_window_wrappers(
    env: gym.Env,
    show_keys: bool = False,
    show_boundary: bool = False,
    show_rays: bool = False
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


WindowWrapperT = Union[VecWindowWrapper, WindowWrapper, VecWindowWrapperSB3]
EnvT = Union[gym.Env, AsyncVectorEnv, VecEnv]

class EnvManager:
    def __init__(self, env_name: str, mode: Literal['play', 'menu', 'movie', 'train'], autoreset: bool = False, ):
        self.base_factory = partial(_make, env_name, mode, autoreset)

    def make(self, movies: list[Optional[Path]], env_modifier: Optional[Callable] = None, vec_class: type[VecEnv] | type[AsyncVectorEnv] = gym.vector.AsyncVectorEnv) -> EnvT:
        factory = env_modifier if env_modifier else self.base_factory

        if len(movies) == 1:
            return cast(gym.Env, factory(movies[-1]))

        part = partial(vec_class, [(lambda m=m: factory(m)) for m in movies])
        if vec_class.__name__ == "SubprocVecEnv":
            vec_env = part(start_method='spawn') # sb3
        elif vec_class.__name__ == "AsyncVectorEnv":
            vec_env = part(context='spawn') # gymnasium
        else:
            raise ValueError(f"Unsupported vector environment class: {vec_class.__name__}")

        return vec_env

    @overload
    def make_windowed(self, movies: list[Optional[Path]], vec_class: type[VecEnv], **kwargs) -> VecWindowWrapperSB3: ...

    @overload
    def make_windowed(self, movies: list[Optional[Path]], vec_class: type[AsyncVectorEnv] = gym.vector.AsyncVectorEnv, **kwargs) -> VecWindowWrapper: ...

    @overload
    def make_windowed(self, movies: Optional[Path], vec_class: type[AsyncVectorEnv], **kwargs) -> WindowWrapper: ...

    def make_windowed(self, movies: list[Optional[Path]] | Optional[Path], scale: int = 1, vec_class: type[VecEnv] | type[AsyncVectorEnv] = gym.vector.AsyncVectorEnv,  **kwargs) -> WindowWrapperT:
        windowed_factory = lambda m: _make_window_wrappers(self.base_factory(m), **kwargs)
        _movies = movies if isinstance(movies, list) else [movies]
        env = self.make(_movies, env_modifier=windowed_factory, vec_class=vec_class)
        return EnvManager.add_window(env, scale)

    @staticmethod
    def add_window(env: EnvT, scale: int = 1) -> WindowWrapperT:
        if isinstance(env, gym.vector.AsyncVectorEnv):
            return VecWindowWrapper(env, scale)
        elif isinstance(env, SubprocVecEnv):
            return VecWindowWrapperSB3(env, scale)
        elif isinstance(env, gym.Env):
            return WindowWrapper(env, scale)
        else:
            raise TypeError("env must be a gym.Env or gym.vector.AsyncVectorEnv")
