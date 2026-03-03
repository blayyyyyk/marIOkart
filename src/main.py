from argparse import ArgumentParser
from gym_mkds.wrappers.window_overlay import MarioKart
from gymnasium.wrappers import FrameStackObservation
import torch
from torch import optim
from gym_mkds.wrappers import (
    VecEnvWindow,
    MoviePlaybackWrapper,
    OverlayWrapper,
    compose_overlays,
)
import multiprocessing
from typing import Optional
import numpy as np
import shutil
from src.utils import sav_to_dsm
import src.models.registry as registry
from src.models.registry import CheckpointState
from gymnasium.vector import AsyncVectorEnv
import gymnasium
import os
from pathlib import Path
from functools import partial
from datetime import datetime
import shutil
from src.train import (
    MarioKartDataset,
    ConcatMarioKartDataset,
    DatasetWrapper,
    prepare_data,
    train_loop,
)
from src.config import *


def _parse_movie(args):
    if args.movie:
        assert os.path.isfile(args.movie), "Movie file is not a file"
        movie_paths = [ROOT_DIR / args.movie]
    elif args.movie_dir:
        movie_dir = ROOT_DIR / args.movie_dir
        assert os.path.isdir(movie_dir), "Movie directory is not a directory"
        movie_paths = list(map(Path, os.listdir(movie_dir)))
        movie_paths = list(map(lambda x: movie_dir / x, movie_paths))

        movie_paths = list(movie_paths)
    else:
        raise ValueError("Either movie or movie_dir must be specified")

    return movie_paths


def collect_dsm(in_path: Path) -> list[Path]:
    movie_paths = []
    if in_path.is_dir():
        for mp in in_path.rglob("*.dsm"):
            movie_paths.append(mp)
    elif in_path.is_file() and in_path.suffix == ".dsm":
        movie_paths.append(in_path)

    return movie_paths


def _record(movie_paths: list[Path], out_paths: list[Path], show: bool = True):
    def create_env(m: Path, o: Path):
        # specify save path for dataset assets
        movie_prefix = m.name.split(".")[0]
        out_path = o / movie_prefix

        # combine several overlays (optional)
        composed_overlays = partial(compose_overlays, funcs=OVERLAYS)

        # build environment
        env = gymnasium.make(
            id="gym_mkds/MarioKartDS-v0",
            rom_path=str(ROM_PATH),
            ray_max_dist=RAY_MAX_DIST,
            ray_count=RAY_COUNT,
        )
        # enable movie playback
        env = MoviePlaybackWrapper(
            env, path=str(m), func=lambda emu: emu.movie.is_playing()
        )
        # enable visual overlay
        env = OverlayWrapper(env, func=composed_overlays)
        # enable dataset recording
        env = DatasetWrapper(env, str(out_path))

        return env
        
        

    movie_paths = list(filter(os.path.isfile, movie_paths))
    assert len(movie_paths) > 0, "No movie files were found"
        
    env = AsyncVectorEnv(
        [(lambda m=m, o=o: create_env(m, o)) for m, o in zip(movie_paths, out_paths)]
    )
    
    window = VecEnvWindow(env)
    
    obs, info = env.reset()

    try:
        print("Starting environment loop. Press Ctrl+C in terminal to exit.")
        while window.is_alive:
            actions = [0] * len(movie_paths)
            obs, reward, terminated, truncated, info = env.step(actions)
            window.update()
            if not np.any(info['movie_playing']):
                window.on_destroy()
        
    except KeyboardInterrupt:
        print("Loop interrupted by user.")
    finally:
        progress = info["race_progress"].tolist()
        for m, p in zip(movie_paths, progress):
            if p >= MIN_PROGRESS_FOR_GOOD_DATASET:
                shutil.copy2(m, PROCESSED_GOOD_DATASET_PATH)
                
        window.close()
        env.close()


def record(args):
    if args.process:
        process(args)
    
    # load paths of backup dsm files
    movie_paths = list(PROCESSED_BAD_DATASET_PATH.rglob("*.dsm"))
    movie_count = len(movie_paths)
    out_paths = [args.dest] * movie_count

    # record in batches
    stride = args.num_proc or movie_count
    for start in range(0, movie_count, stride):
        end = min(start + stride, movie_count)
        mp = movie_paths[start:end]
        op = out_paths[start:end]
        if args.verbose:
            print(f"recorded movies ({start+1} to {end+1} / {movie_count}) ")
        
        p = multiprocessing.Process(target=_record, args=(mp, op))
        p.start()
        p.join() # Wait for the batch to finish entirely
        
        # Catch if the C++ core threw a SIGBUS and crashed the batch
        if p.exitcode != 0:
            print(f"Warning: Batch {start+1}-{end} crashed with exit code {p.exitcode}. Moving to next batch.")

    if args.verbose:
        print(f"{movie_count} movie saved to ")


def train(args):
    
    dat_folders = set([])
    for x in args.source:
        for y in x.rglob('*.dat'):
            dat_folders.add(y.parent)
    data_paths: list[str] = [str(p) for p in dat_folders]
    
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


# TODO
def eval(args):
    movie_paths = collect_dsm(args.source)
    
    
        

    def create_env(m: Path):
        composed_overlays = partial(compose_overlays, funcs=OVERLAYS)
        count = 0
        def stop_func(emu: MarioKart):
            nonlocal count
            if emu.memory.race_ready:
                count += 1
                
            return count < 500
        
        env = gymnasium.make(
            id="gym_mkds/MarioKartDS-v0",
            rom_path=str(ROM_PATH),
            ray_max_dist=RAY_MAX_DIST,
            ray_count=RAY_COUNT,
        )
        env = MoviePlaybackWrapper(env, path=str(m), func=stop_func)
        env = OverlayWrapper(env, func=composed_overlays)
        env = FrameStackObservation(env, stack_size=SEQ_LEN)
        return env

    movie_paths = list(filter(os.path.isfile, movie_paths))
    assert len(movie_paths) > 0, "No movie files were found"

    # Load a registered model architecture
    model = registry.get(
        args.model_name,
        num_features=RAY_COUNT,
        embed_size=EMBED_SIZE,
        embed_count=EMBED_COUNT,
    )
    
    # Initialize the parallel environments
    env = AsyncVectorEnv([(lambda m=m: create_env(m)) for m in movie_paths])
    window = VecEnvWindow(env) # attach vectorized GTK window
    obs, _ = env.reset()


    

    try:
        print("Starting environment loop. Press Ctrl+C in terminal to exit.")
        while window.is_alive:
            with torch.no_grad():
                    tensor_obs = {}
                    for key, val in obs.items():
                        # AsyncVectorEnv creates numpy arrays. Convert to tensor and add to device
                        t = torch.from_numpy(val).to(device)

                        # 2. Apply Normalization (CRITICAL for evaluation)
                        if mdata and key in mdata and "mean" in mdata[key]:
                            # Ensure we are doing float math
                            t = t.to(torch.float32)

                            # Move stats to the correct device
                            mean = torch.tensor(
                                mdata[key]["mean"], device=device, dtype=torch.float32
                            )
                            std = torch.tensor(
                                mdata[key]["std"], device=device, dtype=torch.float32
                            )

                            # Broadcast normalize the batch
                            t = (t - mean) / (std + 1e-8)

                        tensor_obs[key] = t

                    # 3. Forward pass
                    # model expects batched dict: {key: tensor[Batch, Seq, ...]}
                    logits = model(tensor_obs)

                    # 4. Get the predicted actions
                    # Assuming logits shape is [Batch, Num_Classes]
                    _, predicted_classes = torch.max(logits, dim=1)

                    # 5. Convert back to a CPU python list for the environment
                    actions = predicted_classes.cpu().tolist()

            # Step the environments forward with the generated actions
            obs, reward, terminated, truncated, info = env.step(actions)
            window.update()

    except KeyboardInterrupt:
        print("Loop interrupted by user.")
    finally:
        env.close()
        
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
        print(f"movie files successfully processed and copied to {PROCESSED_BAD_DATASET_PATH}")


def main():
    parser = ArgumentParser(
        prog="main.py",
        description="Main entry point for the Mario Kart DS ML visualization application.",
    )
    parser.add_argument(
        "--device",
        help="PyTorch device name (ex. 'cpu', 'mps', 'cuda')",
        type=torch.device,
        default=torch.device(DEFAULT_DEVICE_NAME),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        help="Enable verbose console logging for debugging",
        action="store_true",
    )
    subparsers = parser.add_subparsers(dest="command")

    record_parser = subparsers.add_parser("record", help="Collect model training data.")
    record_parser.add_argument(
        "source",
        nargs="+",
        help="Movie files (.dsm) or Game saves (.sav) to collect observation datasets from. Can either be individual files, a list of files, or a directory of acceptable files.",
        type=Path
    )
    record_parser.add_argument(
        "--dest",
        "-o",
        help="Where to output the datasets to",
        default=INTERIM_DATASET_PATH,
    )
    record_parser.add_argument(
        "--num-proc",
        help="maximum number of subprocesses to spawn",
        default=NUM_PROC
    )
    record_parser.add_argument(
        "--process",
        help="when flag is enabled, will make a call to process() before collecting datasets",
        action="store_true",
    )
    record_parser.set_defaults(func=record)

    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "source",
        nargs="+",
        help="List of training data sources to use",
        type=Path,
    )
    train_parser.add_argument(
        "--model-name",
        help="The name of the model to record a checkpoint of after training",
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
    train_parser.set_defaults(func=train)

    eval_parser = subparsers.add_parser("eval", help="Evaluate a model")
    eval_group = eval_parser.add_mutually_exclusive_group(required=True)
    eval_group.add_argument("--model", help="The model to evaluate")
    eval_group.add_argument(
        "--model-dir", help="The directory of model file paths to evaluate"
    )
    eval_group.add_argument(
        "--user",
        action="store_true",
        help="Enable keyboard input (this mode is for debugging)",
    )
    eval_parser.set_defaults(func=eval)
    process_parser = subparsers.add_parser("process", help="Process movie files")
    process_parser.add_argument(
        "source",
        nargs="+",
        help="Movie files (.dsm) or Game saves (.sav) to collect observation datasets from. Can either be individual files, a list of files, or a directory of acceptable files.",
        type=Path
    )
    process_parser.set_defaults(func=process)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
