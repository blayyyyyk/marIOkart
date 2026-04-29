from gymnasium.envs.registration import register

from mariokart_ml.config import N_KEYS, ROM_PATH, STREAMLIT_UDP_DATA_PORT
from mariokart_ml.environments.env import TimeTrialEnv, TimeTrialObservations
from mariokart_ml.environments.make import EnvManager
from mariokart_ml.wrappers import ControllerAction, TimeTrialReward, WebWrapper

__all__ = [
    "CheckpointOverlay",
    "ControllerAction",
    "ControllerDisplay",
    "ControllerObservation",
    "ProgressReward",
    "RaceStats",
    "RewardDisplayWrapper",
    "SweepingRay",
    "SweepingRayOverlay",
    "TrackBoundary",
    "TimeTrialReward",
    "TimeTrialEnv",
    "TimeTrialObservations",
    "EnvManager",
]

register(
    id="mariokart_ml/TimeTrial-human-v1",
    entry_point="mariokart_ml.environments:TimeTrialEnv",
    additional_wrappers=(ControllerAction.wrapper_spec(n_keys=N_KEYS),),
    kwargs={"rom_path": str(ROM_PATH)},
)

register(
    id="mariokart_ml/TimeTrial-v1",
    entry_point="mariokart_ml.environments:TimeTrialEnv",
    additional_wrappers=(
        TimeTrialObservations.wrapper_spec(),
        TimeTrialReward.wrapper_spec(),
    ),
    kwargs={"rom_path": str(ROM_PATH)},
)

register(
    id="mariokart_ml/TimeTrial-streamlit-v1",
    entry_point="mariokart_ml.environments:TimeTrialEnv",
    additional_wrappers=(
        TimeTrialObservations.wrapper_spec(),
        TimeTrialReward.wrapper_spec(),
        WebWrapper.wrapper_spec(data_port=STREAMLIT_UDP_DATA_PORT),
    ),
    kwargs={"rom_path": str(ROM_PATH)},
)
