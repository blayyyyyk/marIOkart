from gym_mkds.wrappers import (
    CheckpointOverlay,
    ControllerAction,
    ControllerDisplay,
    ControllerObservation,
    ProgressReward,
    RaceStats,
    RewardDisplayWrapper,
    SweepingRay,
    SweepingRayOverlay,
    TrackBoundary,
)
from gymnasium.envs.registration import WrapperSpec, register

from ..config import *
from ..wrappers import ControllerAction, TimeTrialReward
from .env import TimeTrialObservations, TimeTrialEnv
from .make import EnvManager

register(
    id="mariokart_ml/TimeTrial-human-v1",
    entry_point="mariokart_ml.environments:TimeTrialEnv",
    additional_wrappers=(
        ControllerAction.wrapper_spec(n_keys=N_KEYS),
    ),
    kwargs={"rom_path": str(ROM_PATH)}
)

register(
    id="mariokart_ml/TimeTrial-v1",
    entry_point="mariokart_ml.environments:TimeTrialEnv",
    additional_wrappers=(
        TimeTrialObservations.wrapper_spec(),
        TimeTrialReward.wrapper_spec(),
    ),
    kwargs={"rom_path": str(ROM_PATH)},
    max_episode_steps=5000
)

