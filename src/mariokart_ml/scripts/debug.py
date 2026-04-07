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
)
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import Autoreset

from ..config import *
from ..environments import CheckpointReward
from ..environments.boundary_wrapper import BoundaryAngle
from ..environments.reward_wrapper import CumulativeRewardInfo, RewardInfo
from ..models import registry
from ..utils import collect_dsm


def debug(
    mode: Literal["movie", "play"],
    movie_source: list[Path],
    savestate: Optional[int],
    scale: int,
    env_name: str,
):
    if mode == "movie":
        assert movie_source or len(movie_source), "movie source is required for movie-debugging-mode"
        movie_paths = set([])
        for s in movie_source:
            movie_paths |= set(collect_dsm(s))
    elif mode == "play":
        movie_paths = [None]
    else:
        raise ValueError(f"Invalid debug mode provided: {args.mode}")

    def create_env(movie: Optional[Path]):
        env = gymnasium.make(env_name)
        if savestate is not None:
            env = SaveStateWrapper(env, save_slot_id=savestate)

        if movie is not None:
            env = MoviePlaybackWrapper(env, path=str(movie))

        if mode == "play":
            env = HumanInput(env)

        return env

    if mode == "movie":
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
debug_parser.add_argument(
    "mode",
    choices=["movie", "play"],
    help="debugging mode for debug tool. available modes (movie, play)"
)
debug_parser.add_argument(
    "--movie-source",
    help="the file(s) or director(ies) containing movie replays to source menu controls during evaluation (accepted files: .dsm)",
    nargs="+",
    type=Path
)
debug_parser.add_argument("--savestate", "-s", help="Load and save a temporary save state. Press 'o' for saving a state to slot id _ and press 'l' to load a state from slot id _.]", type=int)
debug_parser.set_defaults(func=debug, env="gym_mkds/MarioKartDS-human-v1")


if __name__ == "__main__":
    import os

    from .util import general_parser, script_main, window_parser

    prog = os.path.basename(__file__)
    script_main(prog, [debug_parser, window_parser, general_parser])
