from gym_mkds.wrappers.controller import ControllerObservation
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Literal, Optional

import gymnasium
import numpy as np
from desmume.emulator_mkds import MarioKart
from gym_mkds.wrappers import (
    GtkVecWindow,
    GtkWindow,
    HumanInput,
    MoviePlaybackWrapper,
    RewardDisplayWrapper,
    SaveStateWrapper,
    ControllerDisplay,
)
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import Autoreset
import time

from ..config import *
from ..wrappers.self_driving_reward import SelfDrivingReward, CumulativeRewardInfo, RewardInfo
from ..wrappers.boundary_wrapper import BoundaryAngle
from ..models import registry
from ..utils import collect_dsm

def debug(
    play: Optional[bool],
    movie: Optional[list[Path]],
    scale: int,
    env_name: str,
    show_keys: bool,
    **kwargs
):
    mode = "play" if play else "movie"
    
    if mode == "movie":
        assert movie and len(movie), "movie source is required for movie-debugging-mode"
        movie_paths = set([])
        for s in movie:
            movie_paths |= set(collect_dsm(s))
    elif mode == "play":
        movie_paths = [None]
    else:
        raise ValueError(f"Invalid debug mode provided: {args.mode}")

    def create_env(movie: Optional[Path]):
        env = gymnasium.make(env_name)

        if movie is not None:
            env = MoviePlaybackWrapper(env, path=str(movie))

        if mode == "play":
            env = HumanInput(env)
            
        if show_keys:
            env = ControllerObservation(env, n_keys=N_KEYS)
            env = ControllerDisplay(env, n_physical_keys=N_KEYS)

        return env

    if mode == "movie" and len(movie_paths) > 1:
        env = AsyncVectorEnv(
            [(lambda m=m: create_env(m)) for m in movie_paths]
        )
        env = GtkVecWindow(env, scale)
    elif mode == "play":
        env = create_env(None)
        # env = CheckpointReward(env)
        # env = Autoreset(env)
        # env = TrackBoundary(env)
        # env = BoundaryAngle(env)
        # env = RewardInfo(env)
        # env = RewardDisplayWrapper(env)
        # env = CumulativeRewardInfo(env)
        env = GtkWindow(env, scale)
    elif mode == 'movie' and len(movie_paths) == 1:
        env = create_env(movie_paths.pop())
        env = GtkWindow(env, scale)
    else:
        raise ValueError(f"Invalid debug mode provided: {mode}")

    

    obs, info = env.reset()
    assert env.window is not None

    try:
        print("Starting environment loop. Press Ctrl+C in terminal to exit.")
        while env.window.is_alive:
            actions = [0] * len(movie_paths)
            obs, reward, terminated, truncated, info = env.step(actions)
            
            
            if not mode == "movie": continue
            if not np.any(info["movie_playing"]):
                env.close()

    except KeyboardInterrupt:
        print("Loop interrupted by user.")
    finally:
        env.close()

# Debug Mode Parsing #
debug_parser = ArgumentParser(add_help=False)
mode_group = debug_parser.add_mutually_exclusive_group(required=True)
mode_group.add_argument(
    "--play",
    help="Enable keyboard input",
    action="store_true",
)
mode_group.add_argument(
    "--movie",
    help="the file(s) or director(ies) containing movie replays to source menu controls during evaluation (accepted files: .dsm)",
    nargs="+",
    type=Path
)
debug_parser.add_argument(
    "--show-keys",
    help="Display DS controller overlay",
    action="store_true"
)
debug_parser.set_defaults(func=debug, env_name="mariokart_ml/MarioKartDS-v2")


if __name__ == "__main__":
    import os

    from .util import general_parser, script_main, window_parser

    prog = os.path.basename(__file__)
    script_main(prog, [debug_parser, window_parser, general_parser])
