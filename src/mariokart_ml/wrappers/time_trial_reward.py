from collections.abc import Callable
from functools import cached_property
from typing import Any, cast

import gymnasium as gym
import numpy as np
from desmume.emulator_mkds import MarioKart
from gymnasium.wrappers.utils import RunningMeanStd
from numpy import dot
from numpy.linalg import norm

from mariokart_ml.utils.checkpoint import NKM
from mariokart_ml.wrappers.checkpoint_wrapper import checkpoint_angle_signed


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

        self.cpoi_progress = 0.0
        self.cpoi_id = 0
        self.checkpoint_angle_raw = 0.0

    @cached_property
    def nkm(self) -> NKM:
        emu = cast(MarioKart, self.get_wrapper_attr("emu"))
        return NKM(emu)

    def reset(self, **kwargs):
        self.step_count = 0
        self.snaking_cooldown = 0
        return super().reset(**kwargs)

    def _collision_penalty(self, emu: MarioKart):
        speed = float(emu.memory.driver.speed) / max(float(emu.memory.driver.maxSpeed), 1e-8)

        return -1.0 if speed < 0.4 else 0.0

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
        progress = (drift_left_count + drift_right_count) / 4  # noqa: F841

        reward = 1.0 if drift_direction != 0.0 else 0.0

        weight = 0.2

        # ensure forward progress when drifting
        curr_cpoi_id = emu.memory.race_status.driverStatus[0].curCpoi
        curr_cpoi_progress = float(emu.memory.race_status.driverStatus[0].cpoiProgress)

        backward_progress = curr_cpoi_progress - self.cpoi_progress < 0
        if curr_cpoi_id == self.cpoi_id and backward_progress:
            weight = 0.0

        self.cpoi_id = curr_cpoi_id
        self.cpoi_progress = curr_cpoi_progress

        # ensure pointed towards checkpoint when drifting

        return reward * weight

    def _speed_reward(self, emu: MarioKart):
        speed = float(emu.memory.driver.speed) / max(float(emu.memory.driver.maxSpeed), 1e-8)

        weight = 0.5

        curr_cpoi_id = emu.memory.race_status.driverStatus[0].curCpoi
        curr_cpoi_progress = float(emu.memory.race_status.driverStatus[0].cpoiProgress)

        backward_progress = curr_cpoi_progress - self.cpoi_progress < 0
        if curr_cpoi_id == self.cpoi_id and backward_progress:
            weight = 0.0

        self.cpoi_id = curr_cpoi_id
        self.cpoi_progress = curr_cpoi_progress

        return speed * weight

    def _drift_boost_reward(self, emu: MarioKart):
        drift_direction = float(emu.memory.driver.leftRightDir)
        drift_released = drift_direction != 0 and not self._prev_drift_direction != 0
        self._prev_drift_direction = drift_direction  # update prev drift direction
        drift_boost_active = emu.memory.driver.driftBoostCounter > 20

        drift_reward = 1.0 if drift_released and drift_boost_active else 0.0

        weight = 1.0

        if abs(self.checkpoint_angle_raw) < 0.20:
            weight = 0.0

        correct_timing = abs(checkpoint_angle_signed(emu, direction_mode="movement", lookahead_id=1)) < 0.2
        if drift_released and correct_timing:
            weight += 0.2

        return drift_reward * weight

    def _checkpoint_angle_reward(self, emu: MarioKart):
        # alpha = 2.0

        # alphas = [(1 / alpha**i) for i in range(1, 4)]
        # alpha_sum = sum(alphas)

        # total = 0.0

        # # Compute the sum of weighted checkpoint angles using the exponential decay schedule
        # for i, alpha in enumerate(alphas):
        #     angle = checkpoint_angle_signed(emu, direction_mode="movement", lookahead_id=i+1) # 0=forward, -0to-1=left, +0to+1=right
        #     total += alpha * abs(angle) / alpha_sum

        # # total SHOULD be bounded by [0, 1]

        # weight = 0.3

        # self.checkpoint_angle_raw = total # caching expensive computation
        # reward = 1.0 - total

        # if total > 0.2:
        #     weight = 0.0

        drift_active = float(emu.memory.driver.leftRightDir) != 0
        angle_error = self.nkm.get_checkpoint_angle_error(mid_offset=0.02) / np.pi

        weight = 0.0
        if (drift_active and angle_error < 0.5) or (not drift_active and angle_error < 0.3):
            weight = 1.0

        return 0.2 * weight

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

        # apply rewards
        reward += self.reward_component(emu, self._drifting_reward)
        reward += self.reward_component(emu, self._snaking_reward)
        reward += self.reward_component(emu, self._drifting_direction_reward)
        reward += self.reward_component(emu, self._drift_boost_reward)
        reward += self.reward_component(emu, self._collision_penalty)
        reward += self.reward_component(emu, self._existing_penalty)
        reward += self.reward_component(emu, self._speed_reward)
        reward += self.reward_component(emu, self._checkpoint_angle_reward)

        reward = reward if reward < 2.0 else 2.0  # max clip
        reward = reward if reward > -2.0 else -2.0  # min clip

        return reward
