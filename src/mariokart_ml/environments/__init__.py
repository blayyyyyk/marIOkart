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
from gymnasium.wrappers import Autoreset

from ..config import *
from ..wrappers.boundary_wrapper import BoundaryAngle
from ..wrappers.checkpoint_wrapper import Checkpoint
from ..wrappers.dataset_wrapper import DatasetWrapper
from ..wrappers.state_sampling import SaveStateSampling
from ..wrappers.time_trial_reward import TimeTrialReward
from .env import TimeTrialEnv, TimeTrialObservations

register(
    id="gym_mkds/MarioKartDS-human-v1",
    entry_point="gym_mkds.envs:MarioKartCoreEnv",
    additional_wrappers=(
        SweepingRay.wrapper_spec(n_rays=RAY_COUNT, min_val=0, max_val=RAY_MAX_DIST),
        # RaceStats.wrapper_spec(),
        ControllerObservation.wrapper_spec(n_keys=N_KEYS),
        ControllerAction.wrapper_spec(n_keys=N_KEYS),
        TrackBoundary.wrapper_spec(),
        SweepingRayOverlay.wrapper_spec(color_map=RAY_COLOR_MAP),
        #ControllerDisplay.wrapper_spec(n_physical_keys=N_KEYS),
        Checkpoint.wrapper_spec(),
        CheckpointOverlay.wrapper_spec(),
    ),
    kwargs={"rom_path": str(ROM_PATH)}
)

register(
    id="mariokart_ml/TimeTrial-human-v1",
    entry_point="mariokart_ml.environments:TimeTrialEnv",
    additional_wrappers=(
        ControllerAction.wrapper_spec(n_keys=N_KEYS),
    ),
    kwargs={"rom_path": str(ROM_PATH)}
)

register(
    id="mariokart_ml/TimeTrial-human-v2",
    entry_point="mariokart_ml.environments:TimeTrialEnv",
    additional_wrappers=(
        ControllerAction.wrapper_spec(n_keys=N_KEYS),
        TimeTrialObservations.wrapper_spec(),
        TimeTrialReward.wrapper_spec(),
    ),
    kwargs={"rom_path": str(ROM_PATH)}
)

register(
    id="mariokart_ml/TimeTrial-v1",
    entry_point="mariokart_ml.environments:TimeTrialEnv",
    additional_wrappers=(
        ControllerAction.wrapper_spec(n_keys=N_KEYS),
        TimeTrialObservations.wrapper_spec(),
        TimeTrialReward.wrapper_spec(),
        SaveStateSampling.wrapper_spec(n_samples=20),
        Autoreset.wrapper_spec()
    ),
    kwargs={"rom_path": str(ROM_PATH)}
)
