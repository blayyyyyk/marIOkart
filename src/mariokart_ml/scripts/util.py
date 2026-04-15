from mariokart_ml.utils.vector import get_available_devices
from gymnasium.vector import AsyncVectorEnv
from mariokart_ml.wrappers.window_wrapper import WindowWrapper, VecWindowWrapper
from gym_mkds.wrappers.window import GtkWindow
from gym_mkds.wrappers.human_input import HumanInput
from argparse import ArgumentParser

import gymnasium as gym
import torch
from typing import Literal, Any, Callable, Optional

from ..config import *
from ..environments import *
from gym_mkds.wrappers.window_overlay import ControllerDisplay
from gym_mkds.wrappers.controller import SPARSE_KEYMAP, ControllerRemap, ControllerObservation
from ..wrappers import MovieWrapper
from functools import partial
from warnings import warn


def check_device_cpu(args):
    if not hasattr(args, "device"):
        return
        
    if getattr(args, 'device') == 'cpu':
        warn("CPU device detected, training model performance may be impacted. Consider selecting another device with the `--device` flag.")
        

def script_main(prog, parents: list[ArgumentParser]):
    # parse arguments
    parser = ArgumentParser(prog=prog, parents=parents)
    args = parser.parse_args()
    if hasattr(args, "func"):
        args_list = vars(args)
        args_list.pop('func')
        check_device_cpu(args)
        args.func(**args_list)
    else:
        parser.print_help() # print help if no/invalid mode specified

available_envs = []

for env_id, spec in gym.envs.registry.items():
    if not isinstance(spec.entry_point, str): continue
    if spec.entry_point.startswith('gymnasium.envs'): continue
    available_envs.append(f"{env_id}")




# Window Options Parsing #
window_parser = ArgumentParser(add_help=False)
window_parser.add_argument("--scale", help="specify the scale of the gtk window", type=float, default=1.0)

# General Options Parsing #
general_parser = ArgumentParser(add_help=False)
general_parser.add_argument(
    "--device",
    help=f"Pytorch compute device (Available devices: {get_available_devices()}, Default device: {DEFAULT_DEVICE_NAME})",
    choices=get_available_devices(),
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
    "--env-name",
    choices=available_envs,
    default="gym_mkds/MarioKartDS-human-v1"
)
