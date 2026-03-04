from argparse import ArgumentParser

import torch.optim as optim

from src.config import *
from src.models import registry
from src.models.registry import CheckpointState
from src.train import prepare_data, train_loop
from src.utils import collect_dat
from src.scripts.util import general_parser, window_parser




def train_supervised(args):
    dat_folders = set([])
    for dat_source in args.source:
        dat_folders |= collect_dat(dat_source, args.course_name)

    if args.verbose:
        for folder_path in dat_folders:
            print(f"data source loaded: {folder_path}")

    data_paths = list(map(str, dat_folders))

    # build training data train/test split
    train_split, test_split, mdata = prepare_data(
        data_folders=data_paths,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        split_ratio=args.split_ratio,
    )

    model_dir = CHECKPOINTS_PATH / args.model_name
    ckpt_iter = model_dir.glob("*.ckpt")
    is_empty = not any(ckpt_iter)
    # load most recent checkpoint of specified model
    # or start fresh if there is none
    if not is_empty:
        files = sorted(ckpt_iter, key=lambda f: f.stat().st_mtime, reverse=True)
        model, _ = registry.load(args.model_name, model_dir, args.device)
    else:
        model = registry.get(
            args.model_name,
            num_features=RAY_COUNT,
            embed_size=EMBED_SIZE,
            embed_count=EMBED_COUNT,
        )

    #
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loop(
        model,
        train_split,
        test_split,
        num_epochs=args.epochs,
        optimizer=optimizer,
        device=args.device,
    )

    checkpoint: CheckpointState = {
        "epoch": args.epochs,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None,
        "model_name": args.model_name,
        "mdata": mdata,
        "config": model.get_config(),
    }

    folder = CHECKPOINTS_PATH / f"{args.model_name}"
    registry.save(args.model_name, folder, checkpoint)
    print(f"Checkpoint save in {folder}")


# Training Mode Parsing #
train_parser = ArgumentParser(add_help=False)
train_parser.add_argument(
    "source",
    nargs="+",
    help="List of training data sources to use",
    type=Path,
)
train_parser.add_argument(
    "--model-name",
    help="The name of the model to record a checkpoint of after training",
    default=DEFAULT_MODEL_NAME,
    type=str,
)
train_parser.add_argument(
    "--course-name", "-c", type=str, default=DEFAULT_COURSE_NAME
)
hyper_params_group = train_parser.add_argument_group("Training Hyperparameters")
hyper_params_group.add_argument(
    "--epochs", "-e", help="Number of training epochs.", default=EPOCHS
)
hyper_params_group.add_argument("--lr", help="Learning rate", default=LEARNING_RATE)
hyper_params_group.add_argument(
    "--batch-size", help="Batch size", default=BATCH_SIZE
)
hyper_params_group.add_argument(
    "--split-ratio",
    help="Ratio of training data to split into train and test sets",
    default=SPLIT_RATIO,
)
hyper_params_group.add_argument(
    "--seq-len",
    help="Size of the sequence length of the training samples",
    default=SEQ_LEN,
)
train_parser.set_defaults(func=train_supervised)

def main():
    # parse arguments
    import os
    prog = os.path.basename(__file__)
    parser = ArgumentParser(prog=prog, parents=[train_parser, general_parser])
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help() # print help if no/invalid mode specified

if __name__ == "__main__":
    main()
