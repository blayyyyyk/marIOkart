import multiprocessing as mp
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Optional

import gymnasium
import numpy as np
from gym_mkds.wrappers import (
    MoviePlaybackWrapper,
    OverlayWrapper,
    SaveStateWrapper,
    VecEnvWindow,
    compose_overlays,
)
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import FrameStackObservation
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.config import *
from src.scripts.util import general_parser, window_parser
from src.utils import Suppress, collect_dsm
from src.utils.functional import sub_process_func
from src.utils.recording.extract_ghost import COURSE_ABBREVIATIONS


class WindowUpdateCallback(BaseCallback):
    def __init__(self, window):
        super().__init__()
        self.window = window

    def _on_step(self) -> bool:
        self.window.update()
        return self.window.is_alive


def create_env(id: int, m: Optional[Path]):
    composed_overlays = partial(compose_overlays, funcs=OVERLAYS)


    env = gymnasium.make(
        id="gym_mkds/MarioKartDS-v0",
        rom_path=str(ROM_PATH),
        ray_max_dist=RAY_MAX_DIST,
        ray_count=RAY_COUNT,
    )


    func = lambda env: not env.get_wrapper_attr('emu').memory.race_ready
    if m:
        env = SaveStateWrapper(env, save_slot_id=id)
        env = MoviePlaybackWrapper(env, path=str(m), func=func)
    else:
        env = SaveStateWrapper(env, load_slot_id=id)

    env = OverlayWrapper(env, func=composed_overlays)
    env = FrameStackObservation(env, stack_size=SEQ_LEN)
    return env

def loop_movie(env: AsyncVectorEnv):
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

def loop_train(env: AsyncVectorEnv, args):
    window = VecEnvWindow(env)

    # Apply Torch conversions if your SB3 setup requires it
    # env = NumpyToTorch(env, device=args.device)

    algo_class = ALGO_MAP[args.algo]
    algo_kwargs = ALGO_KWARGS.get(args.algo, {})

    model = algo_class("MultiInputPolicy", env, verbose=int(args.verbose), **algo_kwargs)
    callback = WindowUpdateCallback(window)

    try:
        print("Starting RL training. Press Ctrl+C to exit.")
        # total_timesteps should be passed via args
        model.learn(total_timesteps=args.epochs, callback=callback)
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        window.close()

def train_rl(args):
    movie_paths = set([])
    for s in args.movie_source:
        movie_paths |= set(collect_dsm(s, course_name=args.course))

    with Suppress():
        # use movie files to navigate menus
        movie_paths = list(movie_paths)
        save_state_ids = list(range(len(movie_paths)))
        replay_menu_inputs = sub_process_func(loop_movie, create_env)
        replay_menu_inputs(save_state_ids, movie_paths)

        # begin training at savestate where race is starting
        print(mp.get_start_method(allow_none=True))
        bound_train = partial(loop_train, args=args)
        train_model = sub_process_func(bound_train, create_env, vec_class=SubprocVecEnv)
        none_paths = [None] * len(movie_paths)
        train_model(save_state_ids, none_paths)


train_rl_parser = ArgumentParser(add_help=False)
train_rl_parser.add_argument("movie_source", nargs="+", type=Path, help="movie files to perform menu naviagtion before training")
train_rl_parser.add_argument("algo", choices=ALGO_MAP.keys(), help="training algorithm")
train_rl_parser.add_argument("--epochs", type=int, help="number of training epochs", default=EPOCHS)
train_rl_parser.add_argument("--course", choices=list(map(str.lower, COURSE_ABBREVIATIONS)), type=str)
train_rl_parser.set_defaults(func=train_rl)

def main():
    # parse arguments
    import os
    prog = os.path.basename(__file__)
    parser = ArgumentParser(prog=prog, parents=[train_rl_parser, general_parser, window_parser])
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help() # print help if no/invalid mode specified

if __name__ == "__main__":
    main()
