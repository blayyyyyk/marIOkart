from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import gymnasium
import numpy as np
import torch
from gym_mkds.wrappers import GtkVecWindow, MoviePlaybackWrapper
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import FrameStackObservation
from gymnasium.wrappers.vector import NumpyToTorch

from ..config import *
from ..models import registry
from ..utils import collect_dsm
from .util import general_parser, window_parser


def eval_supervised(args):
    movie_paths = set([])
    for s in args.movie_source:
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
    folder_name = CHECKPOINTS_PATH / args.model_name
    model, mdata = registry.load(
        args.model_name,
        folder_name,
        device=args.device
    )

    # Initialize the parallel environments
    env = AsyncVectorEnv([(lambda m=m: create_env(m)) for m in movie_paths])
    env = NumpyToTorch(env, device=args.device)
    env = GtkVecWindow(env)  # attach vectorized GTK window
    obs, _ = env.reset()
    assert env.window is not None

    try:
        print("Starting environment loop. Press Ctrl+C in terminal to exit.")
        while env.window.is_alive:
            with torch.no_grad():
                tensor_obs = {}
                for key, val in obs.items():
                    t = val.to(args.device)

                    if mdata and key in mdata and "mean" in mdata[key]:
                        t = t.to(torch.float32)
                        mean = torch.tensor(
                            mdata[key]["mean"], device=args.device, dtype=torch.float32
                        )
                        std = torch.tensor(
                            mdata[key]["std"], device=args.device, dtype=torch.float32
                        )
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
eval_parser.add_argument(
    "model_name",
    help="name of model to evaluate",
    type=str,
    default=DEFAULT_MODEL_NAME
)
eval_parser.add_argument(
    "movie_source",
    help="the file(s) or director(ies) containing movie replays to source menu controls during evaluation (accepted files: .dsm)",
    nargs="+",
    type=Path
)
eval_parser.set_defaults(func=eval)

def main():
    # parse arguments
    import os
    prog = os.path.basename(__file__)
    parser = ArgumentParser(prog=prog, parents=[eval_parser, window_parser, general_parser])
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help() # print help if no/invalid mode specified

if __name__ == "__main__":
    main()
