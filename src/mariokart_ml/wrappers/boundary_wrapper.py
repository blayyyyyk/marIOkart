from typing import Literal, cast

import gymnasium as gym
import numpy as np
from desmume.emulator_mkds import MarioKart
from gym_mkds.wrappers.sweeping_ray import (
    find_current_boundary_lines,
    get_standing_triangle_id,
)


def project_2d(p0, p1, x):
    d0 = x - p1
    d1 = p0 - p1
    t = np.vecdot(d0, d1) / np.linalg.norm(d1) ** 2  # projection factor
    p_intersect = p1 + t[:, None] * d1
    return p_intersect


def closest_track_point(
    emu: MarioKart,
    direction_mode: Literal["movement"] | Literal["direction"] = "movement",
    default_collision_type: int = 0,
    eps: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the angle the kart is facing/moving towards from the next checkpoint boundary.

    Args:
        emu: MarioKart emulator instance
        direction_mode: "movement" or "direction" (default: "movement")
    Returns:
        An angle value, (0, -pi) when the checkpoint is to the right, (0, +pi) when checkpoint is to the left of the kart., 0 = the kart is facing the next checkpoint, +/- 1 = the kart is facing away
    """
    kart_position = emu.memory.driver_position

    if direction_mode == "movement":
        kart_mtx = emu.memory.driver_matrix2  # (3, 3)
    elif direction_mode == "direction":
        kart_mtx = emu.memory.driver_matrix  # (3, 3)
    else:
        raise ValueError(f"Invalid direction_mode: {direction_mode}")

    boundary_lines = find_current_boundary_lines(emu, mode="custom", min_id=default_collision_type)  # get the ideal track points
    p0, p1 = boundary_lines[:, 0], boundary_lines[:, 1]
    p_intersect = project_2d(p0, p1, kart_position)

    displacement = p_intersect - kart_position

    dist = np.linalg.norm(displacement, keepdims=True, axis=-1)
    min_id = np.argmin(dist)
    min_dist = dist[min_id]
    min_displacement = displacement[min_id]

    if min_dist < eps:
        return np.zeros((3,), dtype=np.float32), min_dist

    min_displacement /= min_dist
    checkpoint_pts_local = kart_mtx @ min_displacement.T  # project to local space (3, 2)

    angle = np.arctan2(checkpoint_pts_local[0], checkpoint_pts_local[2])  # (2,)
    angle = np.min(angle)

    return angle, min_dist


class BoundaryAngle(gym.ObservationWrapper):
    def __init__(self, env: gym.Env[gym.spaces.Dict, gym.spaces.Box]):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Dict)
        self.observation_space = gym.spaces.Dict(
            {
                **env.observation_space.spaces,
                "track_angle": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(1,)),
                "track_distance": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(1,)),
            }
        )
        self.default_collision_type = None
        self.prev_angle = 0.0  # used for reporting info
        self.prev_dist = 0.0

    def observation(self, observation):
        emu: MarioKart = cast(MarioKart, self.get_wrapper_attr("emu"))
        if not emu.memory.race_ready:
            return {
                **observation,
                "track_angle": np.array([0.0], dtype=np.float32),
                "track_distance": np.array([0.0], dtype=np.float32),
            }

        col_id = get_standing_triangle_id(emu)
        col_type = emu.memory.kcl.prism_attributes["collision_type"][col_id]
        if self.default_collision_type is None:
            self.default_collision_type = col_type
        elif col_type == self.default_collision_type:
            self.prev_angle = 0.0
            self.prev_dist = 0.0
            return {
                **observation,
                "track_angle": np.array([0.0], dtype=np.float32),
                "track_distance": np.array([0.0], dtype=np.float32),
            }

        angle, dist = closest_track_point(emu, default_collision_type=self.default_collision_type)

        self.prev_dist = float(dist.item())
        self.prev_angle = float(angle.item())
        return {
            **observation,
            "track_angle": np.array([angle.item()], dtype=np.float32),
            "track_distance": np.array([dist.item()], dtype=np.float32),
        }

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        info = {
            **info,
            "track_angle": self.prev_angle,
            "track_distance": self.prev_dist,
        }

        return observation, reward, terminated, truncated, info
