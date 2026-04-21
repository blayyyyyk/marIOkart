from argparse import SUPPRESS, ArgumentParser
from warnings import warn

import gymnasium as gym
import torch

from mariokart_ml.config import DEFAULT_DEVICE_NAME
from mariokart_ml.utils.vector import get_available_devices


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

# search for available environment options
available_envs = []
for env_id, spec in gym.envs.registry.items():
    if not isinstance(spec.entry_point, str): continue
    if spec.entry_point.startswith('gymnasium.envs'): continue
    available_envs.append(f"{env_id}")

# Window Options Parsing #
window_parser = ArgumentParser(add_help=False)
window_parser.add_argument("--scale", help="specify the scale of the gtk window", type=float, default=SUPPRESS)

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
)
