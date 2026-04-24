from argparse import SUPPRESS, ArgumentParser
from pathlib import Path
from typing import cast

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.vec_env import SubprocVecEnv

from mariokart_ml.config import (
    ALGO_KWARGS,
    ALGO_MAP,
    TOTAL_TRAINING_TIMESTEPS,
)
from mariokart_ml.environments import EnvManager
from mariokart_ml.wrappers.window_wrapper import (
    VecWindowWrapperSB3,
)


class WindowUpdateCallback(BaseCallback):
    def __init__(self, window):
        super().__init__()
        self.window = window

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        if rewards is not None:
            # For multiple processes, you might want to average them or print the first one
            avg_reward = np.mean(rewards)
            print(f"Step: {self.num_timesteps} | Mean Reward: {avg_reward:.4f}", end="\r")

        if self.window is not None:
            self.window.update()
            return self.window.is_alive
        else:
            return True


SPARSE_KEYMAP = {0: 17, 1: 33, 2: 1}


def train(
    algorithm: str = "ppo",
    total_timesteps: int = TOTAL_TRAINING_TIMESTEPS,
    scale: int = 1,
    num_procs: int = 1,
    verbose: bool = False,
    window: bool = False,
    reuse_save_slots: bool = False,
    env_name: str = "mariokart_ml/TimeTrial-v1",
    sample_from: Path | None = None,
    save_model_path: Path | None = None,
    load_model_path: Path | None = None,
    device: torch.device | str = "cpu",
    **kwargs,
):
    algo_class = ALGO_MAP[algorithm]
    algo_kwargs = ALGO_KWARGS.get(algorithm, {})

    movie_paths = [sample_from] * num_procs
    mgr = EnvManager(
        env_name,
        mode="train",
        autoreset=True,
        reuse_save_slots=reuse_save_slots or not window,
    )

    if window:
        env = mgr.make_windowed(movie_paths, scale=scale, vec_class=SubprocVecEnv)
        assert hasattr(env, "window")

        obs = env.reset()
        assert env.window is not None
        env.window.show_menu = not reuse_save_slots
        callback = WindowUpdateCallback(env.window)
    else:
        env = cast(GymEnv, mgr.make(movie_paths, vec_class=SubprocVecEnv))
        obs = env.reset()
        callback = WindowUpdateCallback(None)

    assert env is not None
    model = algo_class("MultiInputPolicy", env=env, verbose=int(verbose), device=device, **algo_kwargs)
    if load_model_path is not None and load_model_path.exists():
        try:
            model.load(load_model_path)
        except Exception as e:
            print(f"Failed to load model from {load_model_path}: {e}")

    while True:
        if isinstance(env, SubprocVecEnv | VecWindowWrapperSB3):
            assert isinstance(obs, dict)
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, dones, info = env.step(action)

        if window and hasattr(env, "window") and env.window is not None:
            assert isinstance(env, VecWindowWrapperSB3)
            env.window.update()

        all_ready = np.all(dones)
        if all_ready:
            env.env_method("enable")  # type: ignore
            break

    try:
        print("Starting RL training. Press Ctrl+C to exit.")
        # total_timesteps should be passed via args
        model.learn(total_timesteps=total_timesteps, callback=callback)
        if save_model_path is not None:
            model.save(save_model_path)
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        env.close()


train_parser = ArgumentParser(add_help=False)
train_parser.add_argument(
    "--sample-from",
    type=Path,
    help="""movie file that performs menuing and single race lap to collect savestates for sampling position,
    if none is specified, keyboard input will be enable for the menu and first lap before training.""",
    default=SUPPRESS,
)
train_parser.add_argument("--algorithm", choices=ALGO_MAP.keys(), help="training algorithm", default=SUPPRESS)
train_parser.add_argument("--total-timesteps", type=int, help="number of training timesteps", default=SUPPRESS)
train_parser.add_argument(
    "--window",
    "-w",
    action="store_true",
    help="display a window showing the agent's environment",
)
train_parser.add_argument(
    "--reuse-save-slots",
    action="store_true",
    help="reuse save slots for training, if not specified, new save slots will be created for each training session. If running headless, this will be forcibly enabled.",
)
train_parser.add_argument(
    "--num-procs",
    help="Specify the number of processes to debug an emulator on. NOTE: play mode does not support multiple processes",
    type=int,
)
train_parser.add_argument("--save-model-path", type=Path, help="Path to save the trained model")
train_parser.add_argument("--load-model-path", type=Path, help="Path to load a pre-trained model")
train_parser.set_defaults(func=train, env_name="gym_mkds/MarioKartDS-v1")


if __name__ == "__main__":
    import os

    from .util import general_parser, script_main, window_parser

    prog = os.path.basename(__file__)
    parser = script_main(prog, [train_parser, general_parser, window_parser])
