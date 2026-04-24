import json
from argparse import ArgumentParser

from mariokart_ml.scripts.debug import debug_parser
from mariokart_ml.scripts.eval import eval_parser
from mariokart_ml.scripts.process import process_parser
from mariokart_ml.scripts.record import record_parser
from mariokart_ml.scripts.train import train_parser
from mariokart_ml.scripts.train_rl import train_rl_parser
from mariokart_ml.scripts.util import general_parser, window_parser

# intercept the existing parser to accept json files
config_parser = ArgumentParser(add_help=False)
config_parser.add_argument("-c", "--config", type=str, help="Path to a JSON config file")
pre_args, remaining_argv = config_parser.parse_known_args()
json_defaults = {}
if pre_args.config:
    with open(pre_args.config) as f:
        json_defaults = json.load(f)

parser = ArgumentParser(
    prog="main.py",
    description="Main entry point for the Mario Kart DS ML visualization application.",
)

parser.add_argument("-c", "--config", type=str, help="Path to a JSON config file")


general_parser.set_defaults(**json_defaults)
train_rl_parser.set_defaults(**json_defaults)
train_parser.set_defaults(**json_defaults)
debug_parser.set_defaults(**json_defaults)
# ^ add the others per script ...

subparsers = parser.add_subparsers(dest="command")
subparsers.add_parser("record", parents=[record_parser, general_parser, window_parser])
subparsers.add_parser("train", parents=[train_parser, general_parser, window_parser])
subparsers.add_parser("train_rl", parents=[train_rl_parser, general_parser, window_parser])
subparsers.add_parser("eval", parents=[eval_parser, general_parser, window_parser])
subparsers.add_parser("process", parents=[process_parser, general_parser])
subparsers.add_parser("debug", parents=[debug_parser, general_parser, window_parser])

# inject json file contents into parser defaults
args = parser.parse_args(remaining_argv)

if hasattr(args, "func"):
    print(args)
    args_list = vars(args)
    func = args_list.pop("func")
    args_list.pop("command")

    # remove leftover config arg from main parser
    args_list.pop("config", None)

    func(**args_list)
else:
    parser.print_help()  # print help if no/invalid mode specified
