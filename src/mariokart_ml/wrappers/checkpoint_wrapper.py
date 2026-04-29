import math
from typing import Literal

import gymnasium as gym
import numpy as np
from desmume.emulator_mkds import MarioKart


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a /= np.linalg.norm(a, axis=-1, keepdims=True)
    b /= np.linalg.norm(b, axis=-1, keepdims=True)
    return np.sum(a * b, axis=-1)


def checkpoint_angle_unsigned(
    emu: MarioKart,
    direction_mode: Literal["movement"] | Literal["direction"] = "movement",
) -> float:
    """
    Computes the direction the kart is facing toward the two endpoints of the next checkpoint

    Args:
        emu: MarioKart emulator instance
        direction_mode: "movement" or "direction" (default: "movement")
    Returns:
        the minimum cosine similarity of the kart's two displacement vectors, the left and right endpoints of the next checkpoint
    """
    cp_position = emu.memory.checkpoint_info()["current_checkpoint_pos"]
    kart_position = emu.memory.driver_position
    displacement = cp_position - kart_position
    if direction_mode == "movement":
        kart_direction = emu.memory.driver_velocity
    elif direction_mode == "direction":
        kart_direction = emu.memory.driver_direction
    else:
        raise ValueError(f"Invalid direction_mode: {direction_mode}")

    angle = cosine_similarity(kart_direction, displacement)

    return np.min(angle).item()


def get_checkpoint_id(emu: MarioKart, id: int) -> int:
    return int(id % emu.memory.map_data.cpoiCount)


def get_checkpoint_pos(emu: MarioKart, id: int) -> np.ndarray:
    entry = emu.memory.checkpoint_data[id]
    _y = float(emu.memory.camera.target.y)
    p0 = [float(entry.x1), _y, float(entry.z1)]
    p1 = [float(entry.x2), _y, float(entry.z2)]
    return np.array([p0, p1], dtype=np.float32)


def checkpoint_angle_signed(emu: MarioKart, direction_mode: Literal["movement"] | Literal["direction"] = "movement", eps: float = 1e-7, lookahead_id: int = 1) -> float:
    """
    Returns the angle the kart is facing/moving towards from the next checkpoint boundary.

    Args:
        emu: MarioKart emulator instance
        direction_mode: "movement" or "direction" (default: "movement")
    Returns:
        An angle value, (0, -pi) when the checkpoint is to the right, (0, +pi) when checkpoint is to the left of the kart., 0 = the kart is facing the next checkpoint, +/- 1 = the kart is facing away
    """

    checkpoint_id = get_checkpoint_id(emu, emu.memory.race_status.driverStatus[0].curCpoi + lookahead_id)  # by default, this is the next checkpoint for the kart
    checkpoint_pts = get_checkpoint_pos(emu, checkpoint_id)  # (2, 3)
    kart_position = emu.memory.driver_position

    if direction_mode == "movement":
        kart_mtx = emu.memory.driver_matrix2  # (3, 3)
    elif direction_mode == "direction":
        kart_mtx = emu.memory.driver_matrix  # (3, 3)
    else:
        raise ValueError(f"Invalid direction_mode: {direction_mode}")

    p0, p1 = checkpoint_pts[0], checkpoint_pts[1]
    d0 = kart_position - p1
    d1 = p0 - p1
    t = np.dot(d0, d1) / np.linalg.norm(d1) ** 2  # projection factor
    p_intersect = p1 + t * d1

    displacement = p_intersect - kart_position

    dist = np.linalg.norm(displacement, keepdims=True, axis=-1)
    if dist < eps:
        return 0.0

    displacement /= dist
    checkpoint_pts_local = kart_mtx @ displacement.T  # project to local space (3, 2)

    angle = np.arctan2(checkpoint_pts_local[0], checkpoint_pts_local[2])  # (2,)
    angle = np.min(angle).item()
    return float(angle / math.pi)


class Checkpoint(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        if isinstance(self.observation_space, gym.spaces.Dict):
            self.observation_space = gym.spaces.Dict(
                {
                    **self.observation_space.spaces,
                    "checkpoint_angle": gym.spaces.Box(-1, 1, (1,), dtype=np.float32),
                }
            )
        else:
            self.observation_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)

    def observation(self, observation: dict):
        emu: MarioKart = self.get_wrapper_attr("emu")
        if not emu.memory.race_ready:
            return {
                **observation,
                "checkpoint_angle": np.array([0.0], dtype=np.float32),
            }

        checkpoint_angle = checkpoint_angle_signed(emu, direction_mode="movement")
        return {
            **observation,
            "checkpoint_angle": np.array([checkpoint_angle], dtype=np.float32),
        }
