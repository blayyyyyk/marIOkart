from argparse import ArgumentParser
from pathlib import Path

import gymnasium
import torch
from gym_mkds.wrappers import GtkVecWindow, MoviePlaybackWrapper
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import FrameStackObservation
from gymnasium.wrappers.vector import NumpyToTorch

from mariokart_ml.config import CHECKPOINTS_PATH, DEFAULT_MODEL_NAME, SEQ_LEN
from mariokart_ml.models import registry
from mariokart_ml.utils import collect_dsm


def eval_supervised(movie_source: list[Path], model_name: str, device: torch.device):
    movie_paths = set([])
    for s in movie_source:
        movie_paths |= set(collect_dsm(s))

    def create_env(m: Path):
        count = 0

        def stop_func(emu):
            nonlocal count
            if emu.memory.race_ready:
                count += 1

            return count < 500

        env = gymnasium.make("gym_mkds/MarioKartDS-human-v1")
        env = MoviePlaybackWrapper(env, path=str(m), func=stop_func)
        env = FrameStackObservation(env, stack_size=SEQ_LEN)
        return env

    # Load a registered model architecture
    folder_name = CHECKPOINTS_PATH / model_name
    model, mdata = registry.load(model_name, folder_name, device=device)

    # Initialize the parallel environments
    env = AsyncVectorEnv([(lambda m=m: create_env(m)) for m in movie_paths])
    env = NumpyToTorch(env, device=device)
    env = GtkVecWindow(env)  # attach vectorized GTK window
    obs, _ = env.reset()
    assert env.window is not None

    try:
        print("Starting environment loop. Press Ctrl+C in terminal to exit.")
        while env.window.is_alive:
            with torch.no_grad():
                tensor_obs = {}
                for key, val in obs.items():
                    t = val.to(device)

                    if mdata and key in mdata and "mean" in mdata[key]:
                        t = t.to(torch.float32)
                        mean = torch.tensor(mdata[key]["mean"], device=device, dtype=torch.float32)
                        std = torch.tensor(mdata[key]["std"], device=device, dtype=torch.float32)
                        t = (t - mean) / (std + 1e-8)

                    tensor_obs[key] = t

                logits, _ = model(tensor_obs)
                _, actions = torch.max(logits, dim=1)

            # Step the environments forward with the generated actions
            obs, reward, terminated, truncated, info = env.step(actions)

    except KeyboardInterrupt:
        print("Loop interrupted by user.")
    finally:
        env.close()


# Evaluation Mode Parsing #
eval_parser = ArgumentParser(add_help=False)
eval_parser.add_argument("model_name", help="name of model to evaluate", type=str, default=DEFAULT_MODEL_NAME)
eval_parser.add_argument(
    "movie_source",
    help="the file(s) or director(ies) containing movie replays to source menu controls during evaluation (accepted files: .dsm)",
    nargs="+",
    type=Path,
)
eval_parser.set_defaults(func=eval)

if __name__ == "__main__":
    import os

    from .util import general_parser, script_main, window_parser

    prog = os.path.basename(__file__)
    script_main(prog, [eval_parser, window_parser, general_parser])
