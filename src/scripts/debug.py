from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Optional
from desmume.emulator_mkds import MarioKart
from src.scripts.util import general_parser, window_parser
import gymnasium
import numpy as np
from gym_mkds.wrappers import (
    HumanInputWrapper,
    MoviePlaybackWrapper,
    OverlayWrapper,
    EnvWindow,
    SaveStateWrapper,
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
        
        # enable savestate
        if args.savestate is not None:
            env = SaveStateWrapper(env, save_slot_id=args.savestate)
        
        # enable movie playback
        if m:
            env = MoviePlaybackWrapper(
                env, path=str(m)
            )

        # enable visual overlay
        env = compose_overlays(env, *OVERLAYS)

        # enable dataset recording
        if args.mode == "play":
            env = HumanInputWrapper(env)

        return env

    if len(movie_paths) > 1:
        env = AsyncVectorEnv(
            [(lambda m=m: create_env(m)) for m in movie_paths]
        )
        window = VecEnvWindow(env, args.scale)
    else:
        env = create_env(None)
        window = EnvWindow(env, args.scale)

    obs, info = env.reset()

    try:
        print("Starting environment loop. Press Ctrl+C in terminal to exit.")
        while window.is_alive:
            actions = [0] * len(movie_paths)
            obs, reward, terminated, truncated, info = env.step(actions)
            window.update()
            if not isinstance(env, AsyncVectorEnv):
                emu: MarioKart = env.get_wrapper_attr('emu')
                if emu.memory.race_ready:
                    emu.memory.driver.position.add_(500.0)
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