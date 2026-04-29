import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnvWrapper
from torch.utils.tensorboard import SummaryWriter


class TelemetryWrapper(VecEnvWrapper):
    def __init__(self, venv):
        super().__init__(venv)

        # initializes tracking arrays for vectorized environments.
        self._num_envs = venv.num_envs
        self._ep_returns = np.zeros(self._num_envs, dtype=np.float32)
        self._ep_lengths = np.zeros(self._num_envs, dtype=np.int32)
        self._ep_frame_max_reward = np.full(self._num_envs, -np.inf, dtype=np.float32)
        self._ep_component_returns = [{} for _ in range(self._num_envs)]

        # initializes novel racing metric accumulators.
        self._ep_centerline_sq_sum = np.zeros(self._num_envs, dtype=np.float32)
        self._ep_drift_boost_frames = np.zeros(self._num_envs, dtype=np.int32)
        self._ep_max_speed_frames = np.zeros(self._num_envs, dtype=np.int32)
        self._ep_slip_accum = np.zeros(self._num_envs, dtype=np.float32)

        # tracks global running bests across all environments and episodes.
        self._global_max_steps = 0
        self._global_max_frame_reward = -np.inf

    def reset(self):
        # resets all episodic tracking arrays.
        self._ep_returns.fill(0.0)
        self._ep_lengths.fill(0)
        self._ep_frame_max_reward.fill(-np.inf)
        self._ep_component_returns = [{} for _ in range(self._num_envs)]
        self._ep_centerline_sq_sum.fill(0.0)
        self._ep_drift_boost_frames.fill(0)
        self._ep_max_speed_frames.fill(0)
        self._ep_slip_accum.fill(0.0)

        return self.venv.reset()

    def step_wait(self):
        # steps the underlying vectorized environment.
        obs, rewards, dones, infos = self.venv.step_wait()

        assert isinstance(obs, dict)

        # processes transitions for each parallel environment.
        for i in range(self._num_envs):
            self._ep_returns[i] += rewards[i]
            self._ep_lengths[i] += 1

            if rewards[i] > self._ep_frame_max_reward[i]:
                self._ep_frame_max_reward[i] = rewards[i]

            # accumulates reward components injected by the reward wrapper.
            if "reward_components" in infos[i]:
                for k, v in infos[i]["reward_components"].items():
                    self._ep_component_returns[i][k] = self._ep_component_returns[i].get(k, 0.0) + v

            # extracts scalar observations for novel metric calculations.
            centerline_dist = float(obs["centerline_dist"][i][0])
            speed_frac = float(obs["speed_frac"][i][0])
            drift_active = float(obs["drift_boost_active"][i][0])
            facing_angle = float(obs["angle_facing"][i][0])
            inertia_angle = float(obs["angle_inertia"][i][0])

            # accumulates telemetry states.
            self._ep_centerline_sq_sum[i] += centerline_dist**2
            self._ep_drift_boost_frames[i] += 1 if drift_active > 0.0 else 0
            self._ep_max_speed_frames[i] += 1 if speed_frac > 0.95 else 0
            self._ep_slip_accum[i] += abs(facing_angle - inertia_angle)

            # packages terminal episode metrics into the info dictionary.
            if dones[i]:
                self._global_max_steps = max(self._global_max_steps, self._ep_lengths[i])
                self._global_max_frame_reward = max(self._global_max_frame_reward, self._ep_frame_max_reward[i])

                ep_length = self._ep_lengths[i] if self._ep_lengths[i] > 0 else 1

                infos[i]["telemetry"] = {
                    "reward/ep_total": self._ep_returns[i],
                    "reward/ep_average_per_step": self._ep_returns[i] / ep_length,
                    "bests/global_max_steps": self._global_max_steps,
                    "bests/global_max_frame_reward": self._global_max_frame_reward,
                    "bests/ep_max_frame_reward": self._ep_frame_max_reward[i],
                    "metrics/centerline_variance": self._ep_centerline_sq_sum[i] / ep_length,
                    "metrics/throttle_efficiency": self._ep_max_speed_frames[i] / ep_length,
                    "metrics/drift_utilization": self._ep_drift_boost_frames[i] / ep_length,
                    "metrics/average_slip_angle": self._ep_slip_accum[i] / ep_length,
                }

                # appends component breakdown.
                for k, v in self._ep_component_returns[i].items():
                    infos[i]["telemetry"][f"components/{k}_total"] = v
                    infos[i]["telemetry"][f"components/{k}_avg"] = v / ep_length

                # zeroes out environment state for the next episode.
                self._ep_returns[i] = 0.0
                self._ep_lengths[i] = 0
                self._ep_frame_max_reward[i] = -np.inf
                self._ep_component_returns[i].clear()
                self._ep_centerline_sq_sum[i] = 0.0
                self._ep_drift_boost_frames[i] = 0
                self._ep_max_speed_frames[i] = 0
                self._ep_slip_accum[i] = 0.0

        return obs, rewards, dones, infos


class TelemetryCallback(BaseCallback):
    def __init__(self, log_dir: str, num_envs: int, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.writers = []

        # initializes a distinct summary writer for each environment subdirectory.
        for i in range(num_envs):
            env_dir = os.path.join(self.log_dir, f"env_{i}")
            os.makedirs(env_dir, exist_ok=True)
            self.writers.append(SummaryWriter(log_dir=env_dir))

    def _on_step(self) -> bool:
        # iterates over transitions and routes metrics to their respective writers.
        for env_idx, info in enumerate(self.locals.get("infos", [])):
            if "telemetry" in info:
                for key, value in info["telemetry"].items():
                    # writes the scalar using a uniform tag to force overlapping charts.
                    self.writers[env_idx].add_scalar(tag=key, scalar_value=value, global_step=self.num_timesteps)
        return True

    def _on_training_end(self) -> None:
        # flushes buffers and releases file handles upon termination.
        for writer in self.writers:
            writer.flush()
            writer.close()
