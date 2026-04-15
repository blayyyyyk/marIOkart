import json
import os
import numpy as np
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback

# Import your core environment classes here

# Import the mutable components that the LLM agent is allowed to edit
# (We will define the structure of train.py below)
from mariokart_ml.autoresearch.train import AgentRewardWrapper, create_model
import mariokart_ml.environments

class DriftMetricsCallback(BaseCallback):
    """
    Hooks into SB3's training loop to safely extract metrics 
    from the environment without altering the agent's code.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.total_drift_frames = 0
        self.wall_collisions = 0
        self.max_lap_progress = 0.0

    def _on_step(self) -> bool:
        # Access the current observation from the vectorized environment
        obs = self.locals["new_obs"]
        infos = self.locals["infos"]

        # obs is a dict because of gym.spaces.Dict. We check the first env's obs [0]
        if "drift_boost_active" in obs:
            # drift_boost_active is 1.0 if active, -1.0 if not
            if obs["drift_boost_active"][0].item() > 0:
                self.total_drift_frames += 1
        
        # Track max progress from the info dict
        for info in infos:
            if "race_progress" in info:
                self.max_lap_progress = max(self.max_lap_progress, info["race_progress"])

        # (Optional) Add collision tracking if you implement it in info/obs
        if "is_collision" in obs:
            if obs["is_collision"][0].item() > 0:
                self.wall_collisions += 1

        return True

def main():
    env = gym.make("mariokart_ml/MarioKartDS-v3")
    
    
    # 2. Wrap it in the LLM's custom reward function
    env = AgentRewardWrapper(env)
    
    # SB3 requires vectorized environments, even for single instances
    # DummyVecEnv or SubprocVecEnv handles the dict observation spaces well
    from stable_baselines3.common.vec_env import DummyVecEnv
    vec_env = DummyVecEnv([lambda: env])

    # 3. Instantiate the LLM's requested model and hyperparameters
    # Forcing device='mps' or 'cuda' ensures the LLM doesn't accidentally run on CPU
    model = create_model(vec_env)
    
    # 4. Run the experiment
    metrics_callback = DriftMetricsCallback()
    
    try:
        # Lock the experiment to exactly 10,000 steps
        model.learn(total_timesteps=10000, callback=metrics_callback)
        status = "success"
        error_msg = ""
    except Exception as e:
        # Catch errors so the LLM knows if its reward function crashed
        status = "failed"
        raise e
        error_msg = str(e)
    finally:
        vec_env.close()

    # 5. Output the results for the LLM to analyze
    results = {
        "status": status,
        "error_message": error_msg,
        "metrics": {
            "total_drift_frames": metrics_callback.total_drift_frames,
            "max_lap_progress": float(metrics_callback.max_lap_progress),
            "wall_collisions": metrics_callback.wall_collisions
        }
    }

    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Run complete. Status: {status}. Drift Frames: {metrics_callback.total_drift_frames}")

if __name__ == "__main__":
    main()