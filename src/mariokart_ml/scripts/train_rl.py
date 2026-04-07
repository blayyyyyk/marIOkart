from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import torch
import gymnasium as gym
import numpy as np
from desmume.emulator_mkds import MarioKart
from gym_mkds.wrappers import MoviePlaybackWrapper, RewardDisplayWrapper
from gym_mkds.wrappers.controller import ControllerRemap
from gym_mkds.wrappers.window import VecEnvWindow
from gymnasium.wrappers import Autoreset, FrameStackObservation
from gymnasium.wrappers.utils import RunningMeanStd
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.vec_env import SubprocVecEnv

from ..config import *
from ..environments import CheckpointReward
from ..environments.boundary_wrapper import BoundaryAngle
from ..utils import Suppress, collect_dsm
from ..utils.recording.extract_ghost import COURSE_ABBREVIATIONS


class NormalizeDictObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, key: str, epsilon: float = 1e-8):
        super().__init__(env)
        self.key = key
        self.epsilon = epsilon

        # Get the original subspace for the target key
        target_space = self.observation_space.spaces[self.key]

        # Initialize Gymnasium's internal running stats tracker
        self.obs_rms = RunningMeanStd(shape=target_space.shape)

        # Update the bounds of the specific key to [-inf, inf]
        new_spaces = dict(self.observation_space.spaces)
        new_spaces[self.key] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=target_space.shape,
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(new_spaces)

    def observation(self, observation):
        obs_array = observation[self.key]

        # Update the running mean and variance
        self.obs_rms.update(obs_array)

        # Normalize the array
        normalized_array = (obs_array - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)

        # Return a shallow copy of the dict with the updated key
        new_obs = observation.copy()
        new_obs[self.key] = normalized_array

        return new_obs



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

def create_env(r: str, m: Optional[Path]):
    count = 0
    def delay(env: gym.Env):
        nonlocal count
        emu: MarioKart = env.get_wrapper_attr('emu')
        if emu.memory.race_ready:
            count += 1

        return count < 800

    env = gym.make(r)
    env = NormalizeDictObservation(env, key="wall_distances")
    env = Autoreset(env)
    env = MoviePlaybackWrapper(env, path=str(m), func=delay)

    env = ControllerRemap(env, keymap=SPARSE_KEYMAP)
    env = BoundaryAngle(env)
    env = CheckpointReward(env)
    # env = RewardDisplayWrapper(env)
    # env = FrameStackObservation(env, stack_size=SEQ_LEN, padding_type="zero")
    return env


def train_rl(
    movie_source: list[Path],
    algo: str,
    env_name: str,
    epochs: int,
    device: torch.device,
    scale: int,
    course: Optional[str] = None,
    verbose: bool = False
):
    movie_paths = set([])
    for s in movie_source:
        movie_paths |= set(collect_dsm(s, course_name=course))

    algo_class = ALGO_MAP[algo]
    algo_kwargs = ALGO_KWARGS.get(algo, {})

    vec_env = SubprocVecEnv([lambda r=env_name, m=m: create_env(r, m) for m in movie_paths])
    window = VecEnvWindow(vec_env, scale)

    model = algo_class("MultiInputPolicy", vec_env, verbose=int(verbose), **algo_kwargs)
    callback = WindowUpdateCallback(window)
    try:
        print("Starting RL training. Press Ctrl+C to exit.")
        # total_timesteps should be passed via args
        if callback is not None:
            model.learn(total_timesteps=epochs, callback=callback)
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        window.on_destroy()


train_rl_parser = ArgumentParser(add_help=False)
train_rl_parser.add_argument("course", choices=list(map(str.lower, COURSE_ABBREVIATIONS)), type=str)
train_rl_parser.add_argument("--movie_source", nargs="+", type=Path, help="movie files to perform menu naviagtion before training", default=[PROCESSED_GOOD_DATASET_PATH])
train_rl_parser.add_argument("--algo", choices=ALGO_MAP.keys(), help="training algorithm", default="ppo")
train_rl_parser.add_argument("--epochs", type=int, help="number of training epochs", default=EPOCHS)
train_rl_parser.set_defaults(func=train_rl, env_name="gym_mkds/MarioKartDS-human-v1")


if __name__ == "__main__":
    import os

    from .util import general_parser, script_main, window_parser

    prog = os.path.basename(__file__)
    parser = script_main(prog, [train_rl_parser, general_parser, window_parser])
