from argparse import ArgumentParser
from functools import partial
from itertools import batched
from pathlib import Path

import gymnasium
import numpy as np
from gym_mkds.wrappers import (
    ControllerDisplay,
    GtkVecWindow,
    MoviePlaybackWrapper,
    compose_overlays,
)
from gymnasium.vector import AsyncVectorEnv

from ..config import *
from ..environments import DatasetWrapper
from ..utils import Suppress
from ..utils.functional import sub_process_func


def create_env(m: Path, o: Path):
    movie_prefix = m.name.split(".")[0]
    out_path = o / Path(movie_prefix)
    env = gymnasium.make("gym_mkds/MarioKartDS-human-v1")
    env = MoviePlaybackWrapper(env, path=str(m))
    env = DatasetWrapper(env, str(out_path))
    return env

def loop(env: GtkVecWindow):
    obs, info = env.reset()
    assert env.window is not None

    try:
        print("Starting environment loop. Press Ctrl+C in terminal to exit.")
        while env.window.is_alive:
            actions = [0] * env.num_envs
            obs, reward, terminated, truncated, info = env.step(actions)
            if not np.any(info["movie_playing"]):
                env.window.destroy()

    except KeyboardInterrupt:
        print("Loop interrupted by user.")
    finally:
        env.window.destroy()

def record(
    source: list[Path],
    dest: Path,
    num_proc: int,
    verbose: bool = False
):
    # load paths of backup dsm files
    movie_paths = []
    for s in source:
        movie_paths = [*movie_paths, *s.rglob("*.dsm")]

    movie_count = len(movie_paths)
    out_paths = batched([dest] * movie_count, num_proc)
    movie_paths = batched(movie_paths, num_proc)

    # record in batches
    stride = num_proc or movie_count
    _record = sub_process_func(loop, create_env)
    for mp, op in zip(movie_paths, out_paths):
        _record(list(mp), list(op))
        if verbose:
            for m in mp:
                print(f"file {m} recorded")

    if verbose:
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

if __name__ == "__main__":
    import os

    from .util import general_parser, script_main, window_parser
    prog = os.path.basename(__file__)
    script_main(prog, [record_parser, window_parser, general_parser])
