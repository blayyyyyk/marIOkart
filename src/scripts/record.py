from argparse import ArgumentParser
from functools import partial
from itertools import batched
from pathlib import Path

import gymnasium
import numpy as np
from gym_mkds.wrappers import (
    MoviePlaybackWrapper,
    OverlayWrapper,
    VecEnvWindow,
    compose_overlays,
)
from gymnasium.vector import AsyncVectorEnv

from config import *
from src.scripts.util import general_parser, window_parser
from src.train import DatasetWrapper
from src.utils.functional import sub_process_func
from src.utils import Suppress

def create_env(m: Path, o: Path):
    # specify save path for dataset assets
    movie_prefix = m.name.split(".")[0]
    out_path = o / movie_prefix

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
    env = MoviePlaybackWrapper(
        env, path=str(m)
    )
    # enable visual overlay
    env = OverlayWrapper(env, func=composed_overlays)
    # enable dataset recording
    env = DatasetWrapper(env, str(out_path))

    return env

def loop(env: AsyncVectorEnv):
    window = VecEnvWindow(env)

    obs, info = env.reset()

    try:
        print("Starting environment loop. Press Ctrl+C in terminal to exit.")
        while window.is_alive:
            actions = [0] * env.num_envs
            obs, reward, terminated, truncated, info = env.step(actions)
            window.update()
            if not np.any(info["movie_playing"]):
                window.on_destroy()

    except KeyboardInterrupt:
        print("Loop interrupted by user.")
    finally:
        window.close()

def record(args):
    # load paths of backup dsm files
    movie_paths = list(PROCESSED_BAD_DATASET_PATH.rglob("*.dsm"))
    movie_count = len(movie_paths)
    out_paths = batched([args.dest] * movie_count, args.num_proc)
    movie_paths = batched(movie_paths, args.num_proc)

    # record in batches
    stride = args.num_proc or movie_count
    _record = sub_process_func(loop, create_env)
    for mp, op in zip(movie_paths, out_paths):
        _record(list(mp), list(op))
        if args.verbose:
            for m in mp:
                print(f"file {m} recorded")

    if args.verbose:
        print(f"{movie_count} movies saved to ")


# Recording Mode Parsing #
record_parser = ArgumentParser(add_help=False)
record_parser.add_argument(
    "source",
    nargs="+",
    help="Movie files (.dsm) or Game saves (.sav) to collect observation datasets from. Can either be individual files, a list of files, or a directory of acceptable files.",
    type=Path,
)
record_parser.add_argument(
    "--dest",
    "-o",
    help="Where to output the datasets to",
    default=INTERIM_DATASET_PATH,
)
record_parser.add_argument(
    "--num-proc", help="maximum number of subprocesses to spawn", default=NUM_PROC, type=int
)
record_parser.add_argument(
    "--process",
    help="when flag is enabled, will make a call to process() before collecting datasets",
    action="store_true",
)
record_parser.set_defaults(func=record)

def main():
    # parse arguments
    import os
    prog = os.path.basename(__file__)
    parser = ArgumentParser(prog=prog, parents=[record_parser, window_parser, general_parser])
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help() # print help if no/invalid mode specified

if __name__ == "__main__":
    main()
