from src.environments.advanced_observations import AdvancedObservations
from gymnasium.envs.registration import WrapperSpec, register
from gym_mkds.wrappers import CheckpointOverlay, SweepingRay, SweepingRayOverlay, ProgressReward, RaceStats, ControllerObservation, ControllerAction, ControllerDisplay, TrackBoundary, RewardDisplayWrapper
from src.environments.dataset_wrapper import DatasetWrapper
from src.environments.reward_wrapper import CheckpointReward
from src.environments.checkpoint_wrapper import Checkpoint
from src.environments.boundary_wrapper import BoundaryAngle
from src.config import *

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
    id="gym_mkds/MarioKartDS-advanced-v1",
    entry_point="gym_mkds.envs:MarioKartCoreEnv",
    additional_wrappers=(
        ControllerAction.wrapper_spec(n_keys=N_KEYS),
        AdvancedObservations.wrapper_spec(),
    ),
    kwargs={"rom_path": str(ROM_PATH)}
)
