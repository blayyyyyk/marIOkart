import time
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Literal, Optional, cast

import gymnasium
import numpy as np
from desmume.emulator_mkds import MarioKart
from gym_mkds.wrappers import (
    ControllerDisplay,
    GtkVecWindow,
    GtkWindow,
    HumanInput,
    RewardDisplayWrapper,
    SaveStateWrapper,
)
from gym_mkds.wrappers.controller import (
    SPARSE_KEYMAP,
    ControllerObservation,
    ControllerRemap,
)
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import Autoreset

from mariokart_ml.scripts.record import create_env

from ..config import *
from ..environments import EnvManager
from ..models import registry
from ..utils import collect_dsm
from ..wrappers import MovieWrapper
from ..wrappers.boundary_wrapper import BoundaryAngle
from ..wrappers.self_driving_reward import (
    CumulativeRewardInfo,
    RewardInfo,
    SelfDrivingReward,
)
from ..wrappers.window_wrapper import VecWindowWrapper, WindowWrapper


def debug(
    play: Optional[bool],
    movie: Optional[list[Path]],
    menu: Optional[list[Path]],
    scale: int,
    env_name: str,
    show_keys: bool,
    show_boundary: bool,
    show_rays: bool,
    autoreset: bool,
    num_procs: int,
    **kwargs
):
    # argument value protection
    sources: list[Path] = []
    if play:
        mode = "play"
    elif movie:
        mode = "movie"
        assert movie and len(movie), "movie source is required for movie-debugging-mode"
        sources = movie
    elif menu:
        mode = "menu"
        assert menu and len(menu) == 1, "menu source is required for menu-debugging-mode"
        sources = menu
    else:
        raise ValueError(f"Invalid debug mode provided")


    # set movie paths
    if mode in ('menu', 'movie'):
        movie_paths = set([])
        for s in sources:
            movie_paths |= set(collect_dsm(s))
    elif mode == "play":
        movie_paths = [None]
    else:
        raise ValueError(f"Invalid debug mode provided")

    assert mode != 'play' if num_procs > 1 else True, "play mode does not support multiple processes"

    movie_paths: list[Optional[Path]] = list(movie_paths)
    movie_paths *= num_procs

    mgr = EnvManager(env_name, mode, autoreset)
    env = mgr.make_windowed(movie_paths, scale=scale, show_keys=show_keys, show_boundary=show_boundary, show_rays=show_rays)
    obs, info = env.reset()
    assert env.window is not None

    try:
        print("Starting environment loop. Press Ctrl+C in terminal to exit.")
        while env.window.is_alive:
            actions = 0
            if isinstance(env, VecWindowWrapper):
                actions = [0] * len(movie_paths)
            
            obs, reward, terminated, truncated, info = env.step(0)

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
mode_group.add_argument(
    "--menu",
    help="the movie replay exclusively for menuing. will enable keyboard input once race has started. NOTE: you can only provide one file (accepted files: .dsm)",
    nargs="+",
    type=Path
)
debug_parser.add_argument(
    "--show-keys",
    help="Display DS controller overlay",
    action="store_true"
)
debug_parser.add_argument(
    "--show-boundary",
    help="Display DS controller overlay",
    action="store_true"
)
debug_parser.add_argument(
    "--show-rays",
    help="Display DS controller overlay",
    action="store_true"
)
debug_parser.add_argument(
    "--autoreset",
    help="Reset the environment when the kart completes the race",
    action="store_true"
)
debug_parser.add_argument(
    "--num-procs",
    help="Specify the number of processes to debug an emulator on. NOTE: play mode does not support multiple processes",
    type=int,
    default=1
)
debug_parser.set_defaults(func=debug, env_name="mariokart_ml/MarioKartDS-v2")


if __name__ == "__main__":
    import os

    from .util import general_parser, script_main, window_parser

    prog = os.path.basename(__file__)
    script_main(prog, [debug_parser, window_parser, general_parser])
