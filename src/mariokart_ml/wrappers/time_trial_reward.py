from collections.abc import Callable
from typing import Any, cast

import gymnasium as gym
import numpy as np
from desmume.emulator_mkds import MarioKart
from gymnasium.wrappers.utils import RunningMeanStd
from numpy import dot
from numpy.linalg import norm

from mariokart_ml.utils.collision import compute_collision_dists


def cos_sim(a, b):
    return dot(a, b) / (norm(a) * norm(b))


class TimeTrialReward(gym.RewardWrapper):
    def __init__(self, env: gym.Env[dict[str, Any], int]):
        super().__init__(env)
        self.lap_times: list[float] = []
        self.running_speed_stats = RunningMeanStd()
        self.snaking_cooldown = 0
        self._prev_drift_direction = 0
        self.reward_components = {}

    def _collision_penalty(self, emu: MarioKart):
        n_rays = 3

        dist = compute_collision_dists(emu, n_rays=n_rays)

        if dist is None:
            return 0.0

        return -3.0 if np.any(dist < 30.0) else 0.0

    def _existing_penalty(self, emu: MarioKart):
        # life is pain
        return -0.2

    def _drifting_reward(self, emu: MarioKart):
        # reward for drifting, needed for early mt learning
        return 0.5 if emu.memory.driver.driftBoostCounter > 0 else 0.0

    def _drifting_direction_reward(self, emu: MarioKart):
        drift_direction = float(emu.memory.driver.leftRightDir)

        drift_left_count = emu.memory.driver.driftLeftCount  # was: mt_left
        drift_right_count = emu.memory.driver.driftRightCount  # was: mt_right
        progress = (drift_left_count + drift_right_count) / 4

        if progress >= 0.5:
            weight = 1.0
        elif progress >= 0.25:
            weight = 0.0
        else:
            weight = -0.1

        weight = 1.0
        return (0.5 if drift_direction != 0.0 else 0.0) * weight

    def _drifting_penalty(self, emu: MarioKart):
        turning_magnitude = abs(float(emu.memory.driver.turningAmount))
        drift_direction = float(emu.memory.driver.leftRightDir)
        return -0.5 if (turning_magnitude > 0.5 and drift_direction == 0.0) else 0.0

    def _speed_reward(self, emu: MarioKart):
        velocity = np.linalg.norm(emu.memory.driver_velocity)
        cp0, cp1 = np.unstack(emu.memory.checkpoint_pos()["next_checkpoint_pos"])
        checkpoint_pos = (cp0 + cp1) / 2
        pos = emu.memory.driver_position
        diff = checkpoint_pos - pos
        out = cos_sim(diff, emu.memory.driver_velocity) * velocity
        if out < 0.0:
            out = 0.0
        weight = 1.0
        return float(out) * weight

    def _drift_boost_penalty(self, emu: MarioKart):
        drift_direction = float(emu.memory.driver.leftRightDir)
        drift_released = drift_direction != 0 and not self._prev_drift_direction != 0
        self._prev_drift_direction = drift_direction  # update prev drift direction
        drift_boost_active = emu.memory.driver.driftBoostCounter > 0
        return -1.0 if drift_released and not drift_boost_active else 0.0

    def _snaking_reward(self, emu: MarioKart):
        drift_boost_count = emu.memory.driver.driftBoostCounter

        drift_left_count = emu.memory.driver.driftLeftCount  # was: mt_left
        drift_right_count = emu.memory.driver.driftRightCount  # was: mt_right
        drift_progress = (drift_left_count + drift_right_count) / 4

        if self.snaking_cooldown == 0 and drift_boost_count > 0 and drift_progress > 0.99:
            return 1.0
            self.snaking_cooldown = 10
        else:
            return 0.0

    def _surface_grip_penalty(self, emu: MarioKart):
        grip = float(emu.memory.driver.velocityMinusDirMultiplier)
        return -0.5 if grip < 0.5 else 0.0

    def step(self, action: int):
        obs, reward, terminated, truncated, info = super().step(action)

        emu = cast(MarioKart, self.get_wrapper_attr("emu"))
        race_started = self.get_wrapper_attr("race_started")
        if race_started:
            speed = float(emu.memory.driver.speed)
            self.running_speed_stats.update(np.array([speed], dtype=np.float32))
            self.snaking_cooldown = max(self.snaking_cooldown - 1, 0)  # I like this. this is very clean :)

        info["reward_components"] = self.reward_components

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.snaking_cooldown = 0
        return super().reset(**kwargs)

    def reward_component(self, emu, hook: Callable[[MarioKart], float | int]) -> float | int:
        val = hook(emu)
        func_name = getattr(hook, "__name__", str(hook))
        self.reward_components[func_name] = val
        return val

    def reward(self, reward) -> float:
        emu = self.get_wrapper_attr("emu")
        race_started = self.get_wrapper_attr("race_started")
        if not race_started:
            return 0.0

        reward += self.reward_component(emu, self._drifting_reward)
        reward += self.reward_component(emu, self._snaking_reward)
        reward += self.reward_component(emu, self._drifting_penalty)
        reward += self.reward_component(emu, self._drifting_direction_reward)
        reward += self.reward_component(emu, self._drift_boost_penalty)
        reward += self.reward_component(emu, self._surface_grip_penalty)
        reward += self.reward_component(emu, self._collision_penalty)
        reward += self.reward_component(emu, self._existing_penalty)
        reward += self.reward_component(emu, self._speed_reward)

        return reward
