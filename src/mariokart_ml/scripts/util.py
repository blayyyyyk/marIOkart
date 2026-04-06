from argparse import ArgumentParser

import gymnasium as gym
import torch

from ..config import *


available_envs = []

for env_id, spec in gym.envs.registry.items():
    if not isinstance(spec.entry_point, str): continue
    if spec.entry_point.startswith('gymnasium.envs'): continue
    available_envs.append(f" {env_id}")


# Window Options Parsing #
window_parser = ArgumentParser(add_help=False)
window_parser.add_argument("--scale", help="specify the scale of the gtk window", type=float, default=1.0)

# General Options Parsing #
general_parser = ArgumentParser(add_help=False)
general_parser.add_argument(
    "--device",
    help="PyTorch device name (ex. 'cpu', 'mps', 'cuda')",
    type=torch.device,
    default=torch.device(DEFAULT_DEVICE_NAME),
)
general_parser.add_argument(
    "--verbose",
    "-v",
    help="Enable verbose console logging for debugging",
    action="store_true",
)
general_parser.add_argument(
    "--env",
    choices=available_envs
)
