from argparse import ArgumentParser
from gymnasium.wrappers import FrameStackObservation
import torch
from torch import optim
from src.gym.wrapper import (
    OverlayWrapper,
    DatasetWrapper,
    sensor_overlay,
    collision_overlay,
    compose_overlays,
)
from src.gym.env import MarioKartEnv
from src.gym.window import VecEnvWindow
from src.models.registry import get_model
from gymnasium.vector import AsyncVectorEnv
import os
from pathlib import Path
from functools import partial
from datetime import datetime
import shutil
from src.train.run import (
    CheckpointState,
    prepare_data,
    run,
    save_checkpoint,
    load_checkpoint,
)
import json

ROOT_DIR = Path(__file__).resolve().parent.parent
ROM_PATH = ROOT_DIR / "private" / "mariokart_ds.nds"
OVERLAYS = [sensor_overlay, collision_overlay]
DATASET_PATH = ROOT_DIR / "data"
RAW_DATASET_PATH = DATASET_PATH / "raw"  # this is where .dsm input recordings are held
INTERIM_DATASET_PATH = (
    DATASET_PATH / "interim"
)  # this is where collected observation data is held

CHECKPOINTS_PATH = ROOT_DIR / "checkpoints"

RAW_DATASET_PATH.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_PATH.mkdir(parents=True, exist_ok=True)
INTERIM_DATASET_PATH.mkdir(parents=True, exist_ok=True)

RAY_MAX_DIST = 3000
RAY_COUNT = 20

BATCH_SIZE = 16
SEQ_LEN = 32
EMBED_SIZE = 32
EMBED_COUNT = 2048  # number of possible 11-bit keymask
NUM_FEATURES = RAY_COUNT + EMBED_SIZE
EPOCHS = 15
SPLIT_RATIO = 0.8
LEARNING_RATE = 1e-4
DEVICE = "cpu"


def CHECKPOINT_CAPSULE_PATH(name: str, version: int):
    return CHECKPOINTS_PATH / f"{name}_v{str(version)}"


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


def record(args):
    movie_paths = _parse_movie(args)

    def create_env(m: Path):
        base_env = MarioKartEnv(
            rom_path=str(ROM_PATH),
            movie_path=str(m),
            ray_max_dist=RAY_MAX_DIST,
            ray_count=RAY_COUNT,
        )
        composed_overlays = partial(compose_overlays, funcs=OVERLAYS)
        overlay_env = OverlayWrapper(base_env, func=composed_overlays)
        movie_prefix = m.name.split(".")[0]
        raw_path = RAW_DATASET_PATH / m.name
        out_path = INTERIM_DATASET_PATH / movie_prefix

        # backup the movie file
        if not os.path.exists(raw_path):
            shutil.copy(m, raw_path)

        dataset_env = DatasetWrapper(overlay_env, str(out_path))
        return dataset_env

    movie_paths = list(filter(os.path.isfile, movie_paths))
    assert len(movie_paths) > 0, "No movie files were found"
    env = AsyncVectorEnv([(lambda m=m: create_env(m)) for m in movie_paths])
    window = VecEnvWindow(env)
    obs, _ = env.reset()

    try:
        print("Starting environment loop. Press Ctrl+C in terminal to exit.")
        while window.is_alive:
            actions = [0] * len(movie_paths)
            obs, reward, terminated, truncated, info = env.step(actions)
            window.update()
    except KeyboardInterrupt:
        print("Loop interrupted by user.")
    finally:
        env.close()


def train(args):
    train_split, test_split, mdata = prepare_data(
        data_folders=args.data_paths,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        split_ratio=args.split_ratio,
    )
    model = get_model(
        args.model_name,
        num_features=RAY_COUNT,
        embed_size=EMBED_SIZE,
        embed_count=EMBED_COUNT,
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    device = torch.device(args.device)

    run(
        model,
        train_split,
        test_split,
        num_epochs=args.epochs,
        optimizer=optimizer,
        device=device,
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
    save_checkpoint(folder, checkpoint)
    print(f"Checkpoint save in {folder}")


# TODO
def eval(args):
    # Determine if we are driving with a model, or if a user is debugging
    driving_mode = "model" if getattr(args, "model", None) else "user"
    movie_paths = _parse_movie(args)

    def create_env(m: Path):
        base_env = MarioKartEnv(
            rom_path=str(ROM_PATH),
            movie_path=str(m),
            ray_max_dist=RAY_MAX_DIST,
            ray_count=RAY_COUNT,
        )
        composed_overlays = partial(compose_overlays, funcs=OVERLAYS)
        overlay_env = OverlayWrapper(base_env, func=composed_overlays)
        
        # Note: If your obs space is a Dict, ensure FrameStackObservation 
        # handles it correctly, or use a custom dict-stacker.
        stacked_env = FrameStackObservation(overlay_env, stack_size=SEQ_LEN)
        return stacked_env

    movie_paths = list(filter(os.path.isfile, movie_paths))
    assert len(movie_paths) > 0, "No movie files were found"
    
    # Initialize the parallel environments
    env = AsyncVectorEnv([(lambda m=m: create_env(m)) for m in movie_paths])
    window = VecEnvWindow(env)
    
    # AsyncVectorEnv returns a batched observation.
    # If it's a Dict space, obs will be a dict of batched arrays: 
    # e.g., {'image': ndarray[batch, seq, h, w, c], 'speed': ndarray[batch, seq, 1]}
    obs, _ = env.reset()

    model = None
    device = None
    mdata = None

    if driving_mode == "model":
        device = torch.device(getattr(args, "device", "cpu"))
        folder = CHECKPOINTS_PATH / args.model
        # Assume load_checkpoint returns the model and the full checkpoint dict
        model, checkpoint = load_checkpoint(folder, args.model, device=device)
        model.eval()
        
        # Extract the normalization metadata saved during training
        mdata = checkpoint.get("mdata", {})

    try:
        print("Starting environment loop. Press Ctrl+C in terminal to exit.")
        while window.is_alive:
            
            if driving_mode == "user":
                # For debugging, just send 0s or capture actual keyboard input
                actions = [0] * len(movie_paths)
            
            else:
                # --- MODEL INFERENCE ---
                with torch.no_grad():
                    # 1. Convert the batched numpy arrays to PyTorch tensors
                    tensor_obs = {}
                    for key, val in obs.items():
                        # AsyncVectorEnv creates numpy arrays. Convert to tensor and add to device
                        t = torch.from_numpy(val).to(device)
                        
                        # 2. Apply Normalization (CRITICAL for evaluation)
                        if mdata and key in mdata and 'mean' in mdata[key]:
                            # Ensure we are doing float math
                            t = t.to(torch.float32)
                            
                            # Move stats to the correct device
                            mean = torch.tensor(mdata[key]['mean'], device=device, dtype=torch.float32)
                            std = torch.tensor(mdata[key]['std'], device=device, dtype=torch.float32)
                            
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


def main():
    parser = ArgumentParser(
        prog="main.py",
        description="Main entry point for the Mario Kart DS ML visualization application.",
    )
    subparsers = parser.add_subparsers(dest="command")

    record_parser = subparsers.add_parser("record", help="Collect model training data.")
    movie_path_group = record_parser.add_mutually_exclusive_group(required=True)
    movie_path_group.add_argument(
        "--movie", help="The file path of the movie to record a dataset of", type=Path
    )
    movie_path_group.add_argument(
        "--movie-dir",
        help="The directory of movie file paths to record a batch of datasets of",
        type=Path,
    )
    record_parser.set_defaults(func=record)

    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "--model-name",
        help="The name of the model to record a checkpoint of after training",
    )
    train_parser.add_argument(
        "--data-paths",
        nargs="+",
        help="List of training data sources to use",
        type=Path,
    )
    train_parser.add_argument(
        "--device", help="Name of pytorch device to use ('cuda', 'mps', ect.)"
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
    eval_parser.add_argument(
        "--device", help="Name of pytorch device to use ('cuda', 'mps', ect.)"
    )
    eval_group.add_argument(
        "--model-dir", help="The directory of model file paths to evaluate"
    )
    eval_group.add_argument(
        "--user",
        action="store_true",
        help="Enable keyboard input (this mode is for debugging)",
    )
    eval_parser.set_defaults(func=eval)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
