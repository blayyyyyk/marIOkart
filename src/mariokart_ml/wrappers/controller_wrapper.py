import gymnasium as gym
import numpy as np
from typing import Optional
from desmume.emulator_mkds import MarioKart
import pynput
from desmume.controls import Keys, keymask
from ..config import SPARSE_KEYMAP

class ControllerAction(gym.ActionWrapper):
    def __init__(self, env: gym.Env, n_keys: int):
        super().__init__(env)
        self.action_space = gym.spaces.Box(0, n_keys, shape=(1,), dtype=np.uint16)
    
    def action(self, action: Optional[int]) -> int:
        emu: MarioKart = self.get_wrapper_attr('emu')
        emu.input.keypad_update(0)
        if emu.movie.is_playing() or action is None:
            return emu.input.keypad_get()
        elif action is None:
            return 0
        elif isinstance(action, int):
            emu.input.keypad_update(action)
            return action
        elif isinstance(action, (list, np.ndarray)):
            if len(action) == 0:
                action = [0]
            
            emu.input.keypad_update(action[0])
            return action[0]
            
class ControllerObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, n_keys: int):
        super().__init__(env)
        if isinstance(env.observation_space, gym.spaces.Dict):
            self.observation_space = gym.spaces.Dict({
                **env.observation_space.spaces,
                "keymask": gym.spaces.Box(0, n_keys, shape=(1,), dtype=np.uint16),
            })
        else:
            self.observation_space = gym.spaces.Box(0, n_keys, shape=(1,), dtype=np.uint16)
            
    def observation(self, observation):
        emu: MarioKart = self.get_wrapper_attr('emu')
        if isinstance(self.observation_space, gym.spaces.Dict) and isinstance(observation, dict):
            return {
                **observation,
                "keymask": np.array([emu.input.keypad_get()], dtype=np.uint16)
            }
        
        return super().observation(observation)



class ControllerRemap(gym.ActionWrapper):
    def __init__(self, env: gym.Env, keymap: dict[int, int]):
        super().__init__(env)
        self.keymap = keymap
        self.action_space = gym.spaces.Discrete(len(keymap), dtype=np.uint16)
        
    def action(self, action):
        if isinstance(action, int):
            return self.keymap.get(action, 0)
        elif isinstance(action, list):
            if len(action) == 0:
                return 0
            
            return self.keymap.get(action[0], 0)
        
        