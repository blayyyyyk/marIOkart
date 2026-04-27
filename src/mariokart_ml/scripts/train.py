import subprocess
import sys
from argparse import SUPPRESS, ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv

from mariokart_ml.config import ALGO_KWARGS, ALGO_MAP, ROOT_DIR, STREAMLIT_UDP_DATA_PORT, TENSORBOARD_LOG_DIR, TOTAL_TRAINING_TIMESTEPS
from mariokart_ml.environments import EnvManager
from mariokart_ml.wrappers import TelemetryCallback, TelemetryWrapper

SPARSE_KEYMAP = {0: 17, 1: 33, 2: 1}


def warmup_environments(env: VecEnv, show_menu: bool = False):
    obs = env.reset()

    # relies on attribute forwarding through the wrapper stack to access the window.
    if hasattr(env, "window") and env.window is not None:
        env.window.show_menu = show_menu

    while True:
        action = [0] * env.num_envs
        obs, reward, dones, info = env.step(action)

        # delegates update rendering to the unwrapped window component.
        if hasattr(env, "window") and env.window is not None:
            env.window.update()

        if np.all(dones):
            env.env_method("enable")
            return


def start_streamlit(num_instances: int):
    # Construct the command
    # Using sys.executable ensures we use the same Python environment
    app_path = ROOT_DIR / "src" / "mariokart_ml" / "streamlit" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", f"{str(app_path)}", "--", "--num-instances", str(num_instances), "--base-data-port", str(STREAMLIT_UDP_DATA_PORT)]

    # Launch the process
    process = subprocess.Popen(cmd)

    port_list = [STREAMLIT_UDP_DATA_PORT + i for i in range(num_instances)]
    print(f"Streamlit started with PID: {process.pid}\nList of open ports (UDP): {port_list}")
    return process


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
    tensorboard_log_dir: Path = TENSORBOARD_LOG_DIR,
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
        # injects the telemetry wrapper into the windowed environment pipeline.
        env = TelemetryWrapper(env)
        assert hasattr(env, "window")
    else:
        env = cast(VecEnv, mgr.make(movie_paths, vec_class=SubprocVecEnv))
        # injects the telemetry wrapper into the headless environment pipeline.
        env = TelemetryWrapper(env)

    assert env is not None

    # binds the tensorboard logging directory to the model instance.
    model = algo_class("MultiInputPolicy", env=env, verbose=int(verbose), device=device, tensorboard_log=str(tensorboard_log_dir), **algo_kwargs)

    if load_model_path is not None and load_model_path.exists():
        try:
            model.load(load_model_path)
        except Exception as e:
            print(f"Failed to load model from {load_model_path}: {e}")

    try:
        print("Warming up environments...")
        warmup_environments(env, show_menu=(not reuse_save_slots))

        # generates a highly specific, collision-proof run name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"mkds_{algorithm}_{timestamp}"

        # constructs the absolute path for the multi-writer callback.
        base_log_dir = str(Path(tensorboard_log_dir) / run_name)

        # injects the environment count and target directory into the callback.
        telemetry_callback = TelemetryCallback(log_dir=base_log_dir, num_envs=env.num_envs)

        print(f"Starting training phase. Press Ctrl+C to exit. Logging to: {tensorboard_log_dir}/{run_name}")

        # delegates standard trajectory collection and optimization to stable baselines.
        model.learn(total_timesteps=total_timesteps, callback=telemetry_callback, reset_num_timesteps=True)

        if save_model_path is not None:
            model.save(save_model_path)
    except KeyboardInterrupt:
        print("Training interrupted by user.")
        if save_model_path is not None:
            model.save(save_model_path)
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
