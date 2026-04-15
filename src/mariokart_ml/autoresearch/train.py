import gymnasium as gym
from stable_baselines3 import PPO
from typing import Any

class AgentRewardWrapper(gym.RewardWrapper):
    """
    LLM INSTRUCTIONS: Modify this class to shape the reward for drifting.
    You can access the emulator state via self.get_wrapper_attr('emu').
    """
    def __init__(self, env: gym.Env[dict[str, Any], int]):
        super().__init__(env)
        
    def reward(self, reward: float) -> float:
        emu = self.get_wrapper_attr('emu')
        race_started = self.get_wrapper_attr('race_started')
        
        if not race_started:
            return 0.0
            
        # Example starting logic
        drift_count = emu.memory.driver.driftLeftCount + emu.memory.driver.driftRightCount
        speed_frac = float(emu.memory.driver.speed) / float(emu.memory.driver.maxSpeed)
        
        custom_reward = 0.0
        
        # Reward driving fast
        custom_reward += speed_frac * 0.1
        
        # Reward being in a drift state
        if drift_count > 0:
            custom_reward += 1.0
            
        return custom_reward

def create_model(env):
    """
    LLM INSTRUCTIONS: Return a StableBaselines3 model. You may adjust hyperparameters here.
    """
    # Using 'auto' will automatically pick up MPS on Apple Silicon or CUDA
    return PPO(
        "MultiInputPolicy", 
        env, 
        learning_rate=3e-4, 
        n_steps=2048, 
        batch_size=64, 
        verbose=0,
        device="auto" 
    )