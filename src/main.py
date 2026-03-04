from argparse import ArgumentParser
from src.scripts.record import record_parser
from src.scripts.debug import debug_parser
from src.scripts.train import train_parser
from src.scripts.eval import eval_parser
from src.scripts.process import process_parser
from src.scripts.train_rl import train_rl_parser
from src.scripts.util import general_parser, window_parser
import multiprocessing as mp


def main():
    parser = ArgumentParser(
        prog="main.py",
        description="Main entry point for the Mario Kart DS ML visualization application.",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("record", parents=[record_parser, general_parser, window_parser])
    subparsers.add_parser("train", parents=[train_parser, general_parser])
    subparsers.add_parser("train_rl", parents=[train_rl_parser, general_parser, window_parser])
    subparsers.add_parser("eval", parents=[eval_parser, general_parser, window_parser])
    subparsers.add_parser("process", parents=[process_parser, general_parser])
    subparsers.add_parser("debug", parents=[debug_parser, general_parser, window_parser])

    # parse arguments
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help() # print help if no/invalid mode specified


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
