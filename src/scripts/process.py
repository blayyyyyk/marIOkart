from argparse import ArgumentParser
from src.utils import sav_to_dsm
from src.config import *
import shutil
from src.scripts.util import general_parser, window_parser

def process(args):
    # convert any sav files
    sav_to_dsm(args.source, PROCESSED_BAD_DATASET_PATH, args.verbose)

    # backup any dsm files
    shutil.copytree(
        RAW_DATASET_PATH,
        PROCESSED_BAD_DATASET_PATH,
        ignore=shutil.ignore_patterns("*.sav"),
        dirs_exist_ok=True,
    )

    if args.verbose:
        print(
            f"movie files successfully processed and copied to {PROCESSED_BAD_DATASET_PATH}"
        )
        
# Process Mode Parsing #
process_parser = ArgumentParser(add_help=False)
process_parser.add_argument(
    "source",
    nargs="+",
    help="Movie files (.dsm) or Game saves (.sav) to collect observation datasets from. Can either be individual files, a list of files, or a directory of acceptable files.",
    type=Path,
)
process_parser.set_defaults(func=process)

def main():
    # parse arguments
    import os
    prog = os.path.basename(__file__)
    parser = ArgumentParser(prog=prog, parents=[process_parser, general_parser])
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help() # print help if no/invalid mode specified

if __name__ == "__main__":
    main()