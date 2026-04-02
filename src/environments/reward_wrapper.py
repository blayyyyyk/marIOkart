

from gym_mkds.wrappers.sweeping_ray import math
from desmume.emulator_mkds import MarioKart
import numpy as np
import gymnasium as gym
from src.environments.checkpoint_wrapper import checkpoint_angle_signed

class CheckpointReward(gym.RewardWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.prev_angle = 0.0
        self.prev_dist = 0.0
        self.prev_cp_progress = 0.0
        self.running_avg_progress = 0.0
        self.total_progress = 0.0
        self.n_steps = 0

    def _angle_reward(self, emu: MarioKart) -> float:
        return 2 * (0.5 - abs(checkpoint_angle_signed(emu, direction_mode="movement"))) # interval: (0, 1)

    def _dist_reward(self, emu: MarioKart) -> float:
        if not emu.memory.race_ready: return 0.0
        checkpoint_pos = emu.memory.checkpoint_info()["next_checkpoint_pos"]
        p0, p1 = checkpoint_pos[0], checkpoint_pos[1]
        midpoint = (p0 + p1) / 2
        kart_position = emu.memory.driver_position
        midpoint_dist = np.linalg.norm(kart_position - midpoint)
        return (1 / (1 + midpoint_dist)).item()

    def _dist_reward_rate(self, emu: MarioKart) -> float:
        dist = self._dist_reward(emu)
        delta = dist - self.prev_dist
        self.prev_dist = dist
        return -delta

    def _angle_reward_rate(self, emu: MarioKart) -> float:
        angle = self._angle_reward(emu)
        delta = angle - self.prev_angle
        self.prev_angle = angle
        return -delta
        
    def _checkpoint_progress_reward(self, emu: MarioKart) -> float:
        if not emu.memory.race_ready:
            return 0.0
            
        curr_cp_progress = float(emu.memory.race_status.driverStatus[0].cpoiProgress)
        delta = curr_cp_progress - self.prev_cp_progress
        self.prev_cp_progress = curr_cp_progress
        return delta # +delta = right direction, -delta = wrong direction

    def reward(self, reward):
        emu: MarioKart = self.get_wrapper_attr('emu')
        
        new_reward = 0.0
        if emu.memory.race_ready:
            prog = self._checkpoint_progress_reward(emu)
            
            r1 = (1.0 if self._angle_reward(emu) > 0.0 else -1.0) * 0.8
            r3 = (1.0 if prog - 0.008 >= 0.0 else -1.0) * 0.7
            dr1 = (1.0 if self._angle_reward_rate(emu) - 0.1 > 0.0 else -1.0) * 0.1
            dr2 = (1.0 if self._dist_reward_rate(emu) - 0.1 > 0.0 else -1.0) * 0.1
            new_reward = (
                r1 + dr1 + r3 + dr2
            ) + float(reward)
            
            
                
            self.total_progress += prog
            self.n_steps += 1
            self.running_avg_progress = self.total_progress / self.n_steps
            
            if new_reward < -0.1:
                if self.has_wrapper_attr('hp'):
                    hp = self.get_wrapper_attr('hp')
                    self.set_wrapper_attr('hp', hp + new_reward)
        
        return new_reward
        
        
class RewardInfo(gym.Wrapper):
    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        
        info = {
            **info,
            "reward": reward
        }
        
        return observation, reward, terminated, truncated, info
        
class CumulativeRewardInfo(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.total_reward = 0.0
    
    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        
        self.total_reward += float(reward)
        info = {
            **info,
            "total_reward": self.total_reward
        }
        
        return observation, reward, terminated, truncated, info