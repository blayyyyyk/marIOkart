import gymnasium as gym


class WebWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        return obs, reward, done, truncated, info
