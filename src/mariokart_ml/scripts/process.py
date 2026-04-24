import shutil
from argparse import ArgumentParser
from pathlib import Path

from ..config import PROCESSED_BAD_DATASET_PATH, RAW_DATASET_PATH
from ..utils import sav_to_dsm
from .util import general_parser


def process(source: Path | list[Path], verbose: bool = False):
    # convert any sav files
    sav_to_dsm(source, PROCESSED_BAD_DATASET_PATH, verbose)

    # backup any dsm files
    shutil.copytree(
        RAW_DATASET_PATH,
        PROCESSED_BAD_DATASET_PATH,
        ignore=shutil.ignore_patterns("*.sav"),
        dirs_exist_ok=True,
    )

    if verbose:
        print(f"movie files successfully processed and copied to {PROCESSED_BAD_DATASET_PATH}")


# Process Mode Parsing #
process_parser = ArgumentParser(add_help=False)
process_parser.add_argument(
    "source",
    nargs="+",
    help="Movie files (.dsm) or Game saves (.sav) to collect observation datasets from. Can either be individual files, a list of files, or a directory of acceptable files.",
    type=Path,
)
process_parser.set_defaults(func=process)

if __name__ == "__main__":
    import os

    from .util import general_parser, script_main

    prog = os.path.basename(__file__)
    parser = script_main(prog=prog, parents=[process_parser, general_parser])
