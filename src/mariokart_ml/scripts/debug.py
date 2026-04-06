from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Optional

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
from .util import general_parser, window_parser


def debug(args):
    if args.mode == "movie":
        assert args.movie_source or len(args.movie_source), "movie source is required for movie-debugging-mode"
        movie_paths = set([])
        for s in args.movie_source:
            movie_paths |= set(collect_dsm(s))
    elif args.mode == "play":
        movie_paths = [None]
    else:
        raise ValueError(f"Invalid debug mode provided: {args.mode}")

    def create_env(env_name: str, movie: Optional[Path]):
        env = gymnasium.make(env_name)
        if args.savestate is not None:
            env = SaveStateWrapper(env, save_slot_id=args.savestate)

        if movie is not None:
            env = MoviePlaybackWrapper(env, path=str(movie))

        if args.mode == "play":
            env = HumanInput(env)

        return env

    if args.mode == "movie":
        env = AsyncVectorEnv(
            [(lambda e=args.env_name, m=m: create_env(e, m)) for m in movie_paths]
        )
        env = GtkVecWindow(env, args.scale)
    elif args.mode == "play":
        env = create_env(args.env_name, None)
        # env = CheckpointReward(env)
        # env = Autoreset(env)
        # env = TrackBoundary(env)
        # env = BoundaryAngle(env)
        # env = RewardInfo(env)
        # env = RewardDisplayWrapper(env)
        # env = CumulativeRewardInfo(env)
        env = GtkWindow(env, args.scale)
    else:
        raise ValueError(f"Invalid debug mode provided: {args.mode}")

    obs, info = env.reset()
    assert env.window is not None

    try:
        print("Starting environment loop. Press Ctrl+C in terminal to exit.")
        while env.window.is_alive:
            actions = [0] * len(movie_paths)
            obs, reward, terminated, truncated, info = env.step(actions)

            if not args.mode == "movie": continue
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

def main():
    # parse arguments
    import os
    prog = os.path.basename(__file__)
    parser = ArgumentParser(prog=prog, parents=[debug_parser, window_parser, general_parser])
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help() # print help if no/invalid mode specified

if __name__ == "__main__":
    main()
