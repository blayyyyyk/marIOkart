from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Optional

from gym_mkds.wrappers.window import VecEnvWindow
import gymnasium as gym
import numpy as np
from gym_mkds.wrappers import GtkVecWindow, MoviePlaybackWrapper, SaveStateWrapper
from gymnasium.wrappers import FrameStackObservation
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from desmume.emulator_mkds import MarioKart
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


def create_env(m: Optional[Path]):
    count = 0
    def delay(env: gym.Env):
        nonlocal count
        emu: MarioKart = env.get_wrapper_attr('emu')
        if emu.memory.race_ready:
            count += 1
        
        return count < 800
    
    env = gym.make("gym_mkds/MarioKartDS-human-v1")
    env = MoviePlaybackWrapper(env, path=str(m), func=delay)
    env = FrameStackObservation(env, stack_size=SEQ_LEN, padding_type="zero")
    return env

def train_rl(args):
    movie_paths = set([])
    for s in args.movie_source:
        movie_paths |= set(collect_dsm(s, course_name=args.course))

    algo_class = ALGO_MAP[args.algo]
    algo_kwargs = ALGO_KWARGS.get(args.algo, {})
    
    vec_env = SubprocVecEnv([lambda m=m: create_env(m) for m in movie_paths])
    window = VecEnvWindow(vec_env)

    model = algo_class("MultiInputPolicy", vec_env, verbose=int(args.verbose), **algo_kwargs)
    callback = WindowUpdateCallback(window)
    try:
        print("Starting RL training. Press Ctrl+C to exit.")
        # total_timesteps should be passed via args
        if callback is not None:
            model.learn(total_timesteps=args.epochs, callback=callback)
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        window.on_destroy()
    


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
