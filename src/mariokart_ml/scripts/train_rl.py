from argparse import ArgumentParser
from pathlib import Path
from typing import Optional, Union, cast
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from torch import device

from mariokart_ml.wrappers.window_wrapper import VecWindowWrapper, WindowWrapper, VecWindowWrapperSB3

from ..config import ALGO_KWARGS, ALGO_MAP, EPOCHS, PROCESSED_GOOD_DATASET_PATH
from ..environments import EnvManager
from ..utils import collect_dsm
from ..utils.sav_to_dsm import COURSE_ABBREVIATIONS


class WindowUpdateCallback(BaseCallback):
    def __init__(self, window):
        super().__init__()
        self.window = window

    def _on_step(self) -> bool:
        self.window.update()
        return self.window.is_alive


SPARSE_KEYMAP = {
    0: 17,
    1: 33,
    2: 1
}

def train_rl(
    env_name: str,
    algorithm: str,
    epochs: int,
    scale: int,
    num_procs: int,
    verbose: bool = False,
    window: bool = False,
    sample_from: Optional[Path] = None,
    **kwargs
):
    algo_class = ALGO_MAP[algorithm]
    algo_kwargs = ALGO_KWARGS.get(algorithm, {})

    movie_paths = [sample_from] * num_procs
    mgr = EnvManager(env_name, "train", autoreset=True)

    callback = None
    if window:
        env = mgr.make_windowed(movie_paths, scale=scale, vec_class=SubprocVecEnv)
        assert hasattr(env, 'window')
        
        obs = env.reset()
        assert env.window is not None
        env.window.show_menu = True
        callback = WindowUpdateCallback(env.window)
    else:
        env = cast(GymEnv, mgr.make(movie_paths, vec_class=SubprocVecEnv))

    assert env is not None
    model = algo_class("MultiInputPolicy", env=env, verbose=int(verbose), **algo_kwargs)
    while True:
        if isinstance(env, (SubprocVecEnv, VecWindowWrapperSB3)):
            dummy_actions = np.array([env.action_space.sample() for _ in range(env.num_envs)])
            obs, rewards, dones, infos = env.step(dummy_actions)
        
        if window and hasattr(env, 'window') and env.window is not None:
            assert isinstance(env, (WindowWrapper, VecWindowWrapperSB3))
            env.window.update()
            
        all_ready = np.all(dones)
        if all_ready:
            break

    try:
        print("Starting RL training. Press Ctrl+C to exit.")
        # total_timesteps should be passed via args
        model.learn(total_timesteps=epochs, callback=callback)
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        if isinstance(env, (WindowWrapper, VecWindowWrapper)):
            assert env.window is not None
            env.window.destroy()


train_rl_parser = ArgumentParser(add_help=False)
train_rl_parser.add_argument("--sample-from", type=Path, help="movie file that performs menuing and single race lap to collect savestates for sampling position, if none is specified, keyboard input will be enable for the menu and first lap before training.")
train_rl_parser.add_argument("--algorithm", choices=ALGO_MAP.keys(), help="training algorithm", default="ppo")
train_rl_parser.add_argument("--epochs", type=int, help="number of training epochs", default=EPOCHS)
train_rl_parser.add_argument("--window", "-w", action="store_true", help="display a window showing the agent's environment")
train_rl_parser.add_argument(
    "--num-procs",
    help="Specify the number of processes to debug an emulator on. NOTE: play mode does not support multiple processes",
    type=int,
    default=1
)
train_rl_parser.set_defaults(func=train_rl, env_name="gym_mkds/MarioKartDS-v1")


if __name__ == "__main__":
    import os

    from .util import general_parser, script_main, window_parser

    prog = os.path.basename(__file__)
    parser = script_main(prog, [train_rl_parser, general_parser, window_parser])
