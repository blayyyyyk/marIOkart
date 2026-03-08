from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Optional
from src.scripts.util import general_parser, window_parser
import gymnasium
import numpy as np
from gym_mkds.wrappers import (
    HumanInputWrapper,
    MoviePlaybackWrapper,
    OverlayWrapper,
    VecEnvWindow,
    compose_overlays,
)
from gymnasium.vector import AsyncVectorEnv

from src.config import *
from src.models import registry
from src.utils import collect_dsm


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

    def create_env(m: Optional[Path]):
        # combine several overlays (optional)
        composed_overlays = partial(compose_overlays, funcs=OVERLAYS)

        # build environment
        env = gymnasium.make(
            id="gym_mkds/MarioKartDS-v0",
            rom_path=str(ROM_PATH),
            ray_max_dist=RAY_MAX_DIST,
            ray_count=RAY_COUNT,
        )
        # enable movie playback
        if m:
            env = MoviePlaybackWrapper(
                env, path=str(m)
            )

        # enable visual overlay
        env = OverlayWrapper(env, func=composed_overlays)

        # enable dataset recording
        if args.mode == "play":
            env = HumanInputWrapper(env)

        return env

    env = AsyncVectorEnv(
        [(lambda m=m: create_env(m)) for m in movie_paths]
    )

    window = VecEnvWindow(env)

    obs, info = env.reset()

    try:
        print("Starting environment loop. Press Ctrl+C in terminal to exit.")
        while window.is_alive:
            actions = [0] * len(movie_paths)
            obs, reward, terminated, truncated, info = env.step(actions)
            window.update()
            if not args.mode == "movie": continue
            if not np.any(info["movie_playing"]):
                window.on_destroy()

    except KeyboardInterrupt:
        print("Loop interrupted by user.")
    finally:
        window.close()
        env.close()

# Debug Mode Parsing #
debug_parser = ArgumentParser(add_help=False)
debug_parser.add_argument(
    "mode",
    help="debugging mode for debug tool. available modes (movie, play)"
)
debug_parser.add_argument(
    "--movie-source",
    help="the file(s) or director(ies) containing movie replays to source menu controls during evaluation (accepted files: .dsm)",
    nargs="+",
    type=Path
)
debug_parser.set_defaults(func=debug)

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