from pathlib import Path
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.base_class import BaseAlgorithm

from typing import Literal, Any

# Default paths for folders and files #
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ROM_PATH = ROOT_DIR / "private" / "mariokart_ds.nds"

DATASET_PATH = ROOT_DIR / "data"
RAW_DATASET_PATH = DATASET_PATH / "raw"  # this is where .dsm input recordings are held
INTERIM_DATASET_PATH = (
    DATASET_PATH / "interim"
)  # this is where collected observation data is held
PROCESSED_GOOD_DATASET_PATH = DATASET_PATH / "processed_good"
PROCESSED_BAD_DATASET_PATH = DATASET_PATH / "processed_bad"

CHECKPOINTS_PATH = ROOT_DIR / "checkpoints"
RAW_DATASET_PATH.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_PATH.mkdir(parents=True, exist_ok=True)
INTERIM_DATASET_PATH.mkdir(parents=True, exist_ok=True)
PROCESSED_GOOD_DATASET_PATH.mkdir(parents=True, exist_ok=True)
PROCESSED_BAD_DATASET_PATH.mkdir(parents=True, exist_ok=True)

# Default environment params #
RAY_MAX_DIST = 3000  # caps the farthest a ray can extend. prevents rays that don't intersect the map from saturating the model input
RAY_COUNT = 20  # number of rays extending from the kart's position, outward, along the positive half of the kart's local xz-plane
N_KEYS = 12
RAY_COLOR_MAP = "viridis"
SAVE_STATE_SAMPLE_COUNT = 20

# Default training params #
BATCH_SIZE = (
    128  # number of training/testing examples processed by the model in parallel
)
SEQ_LEN = 256  # amount of context to give to the model for decision-making
EMBED_SIZE = 128  # number of embedding dims for the keymask embeddings
EMBED_COUNT = 2**N_KEYS # number of possible 11-bit keymask
NUM_FEATURES = (
    RAY_COUNT + EMBED_SIZE
)  # number of dims of the feature vector fed into the model. part of this will always be the keymask embed
EPOCHS = 100  # number of training epochs for a training session
SPLIT_RATIO = (
    0.8  # ratio of training examples to total dataset examples provided for training
)
LEARNING_RATE = (
    1e-4  # this will likely need to be tweaked if not using the Adam optimizer
)

# If you change this, when you don't specify a device name in the cli, pytorch will use this device instead. It would be wise to keep this as 'cpu'
DEFAULT_DEVICE_NAME = "cpu"
DEFAULT_MODEL_NAME = "lstm"
DEFAULT_COURSE_NAME = "f8c"
NUM_PROC = 9

MIN_PROGRESS_FOR_GOOD_DATASET = 0.9

ALGO_MAP: dict[str, type[BaseAlgorithm]] = {
    "ppo": PPO,
    "dqn": DQN,
    "a2c": A2C
}

ALGO_KWARGS: dict[str, dict[str, Any]] = {
    "ppo": {"n_steps": 2048, "learning_rate": 3e-4},
    "dqn": {"buffer_size": 10000, "learning_rate": 1e-3},
}

SPARSE_KEYMAP = {
    0: 17,
    1: 33,
    2: 0,
    3: 1
}