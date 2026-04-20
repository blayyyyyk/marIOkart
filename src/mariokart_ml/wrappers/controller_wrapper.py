from typing import Literal, Optional

import gymnasium as gym
import numpy as np
import pynput
from desmume.controls import Keys, keymask
from desmume.emulator_mkds import MarioKart

from ..config import SPARSE_KEYMAP
from ..utils.game_event import Event


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
    def __init__(self, env: gym.Env, keymap: dict[int, int], enable_event: Optional[Event] = None):
        super().__init__(env)
        self.keymap = keymap
        self.action_space = gym.spaces.Discrete(len(keymap), dtype=np.uint16)
        self.enabled = False
        self.enable_event = enable_event

    def action(self, action):
        if self.enable_event is not None and self.enable_event.update(self.env):
            self.enabled = True
            self.env.reset()
            print(self.enabled)

        if not self.enabled:
            return np.uint16(action)

        return np.uint16(self.keymap.get(int(action), 0))
        
        
class ControllerRemap(gym.ActionWrapper):
    def __init__(self, env: gym.Env, keymap: dict[int, int]):
        super().__init__(env)
        self.keymap = keymap
        self.action_space = gym.spaces.Discrete(len(keymap), dtype=np.uint16)
        self.enabled = False
        
    def enable(self):
        """Explicitly enables the AI action mapping."""
        self.enabled = True

    def action(self, action):
        if not self.enabled:
            return np.uint16(action)

        return np.uint16(self.keymap.get(int(action), 0))


class ControllerDriftingRemap(ControllerRemap):
    def __init__(self, env: gym.Env):
        super().__init__(
            env,
            keymap={
                0: 1, # forward
                1: 17, # right
                2: 33, # left
                3: 257, # drift forward + toggle drift (on)
                4: 1, # forward + toggle drift (off)
            },
        )
        self.drifting = False

    def action(self, action):
        if not self.enabled:
            return np.uint16(action)

        if action == 3:
            self.drifting = True
        elif action == 4:
            self.drifting = False
            
        keymask = self.keymap.get(int(action), 1)
        if self.drifting:
            keymask |= 256

        return np.uint16(keymask)