from argparse import ArgumentParser
from src.config import *
import torch

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