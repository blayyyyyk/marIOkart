from mariokart_ml.utils.update_rule import Event
import random
from typing import cast, Optional

import gymnasium as gym
import numpy as np
from desmume.emulator_mkds import MarioKart


class SaveStateSampling(gym.Wrapper):
    def __init__(self, env: gym.Env, n_samples: int | list[float], reuse_slots: bool = False, collect_saves_event: Optional[Event] = None):
        super().__init__(env)

        self.reset_event = self.get_wrapper_attr('reset_event') if collect_saves_event is None else collect_saves_event

        if isinstance(n_samples, int):
            self.sample_times = np.linspace(0, 1, n_samples)
        elif isinstance(n_samples, list):
            self.sample_times = np.array(n_samples)

        self.saved_slots: list[int] = []
        self._prev_min_dist = 1.0

        self.create_states = not reuse_slots

    def reset(self, *, seed=None, options=None):
        if len(self.saved_slots) == 0:
            return super().reset(seed=seed)

        slot = random.choice(self.saved_slots)
        print(f"Resetting to saved slot: {slot}")
        return super().reset(seed=seed, options={ "reset_type": slot })

    def _make_state_sample(self):
        emu = cast(MarioKart, self.get_wrapper_attr('emu'))
        race_started = self.get_wrapper_attr('race_started')
        
        if not race_started:
            return

        reset_event_name = self.reset_event.__class__.__name__
        if reset_event_name == "RaceEndEvent":
            progress = float(emu.memory.race_status.driverStatus[0].raceProgress)
        elif reset_event_name == "LapEndEvent":
            progress = float(emu.memory.race_status.driverStatus[0].lapProgress)
        else:
            raise ValueError(f"Invalid reset trigger provided: {self.reset_event}")

        dist = np.abs(progress - self.sample_times)
        current_id = int(dist.argmin().item()) # including offset of GAME_SAVE_SLOT and RACE_SAVE_SLOT

        if current_id not in self.saved_slots:
            emu.savestate.save(current_id + 2)
            self.saved_slots.append(current_id)
            print(f"Save state saved to slot {current_id}")

        if len(self.saved_slots) == self.sample_times.shape[0]:
            self.create_states = False

    def step(self, action):
        event_state = self.reset_event.update(self.env) if self.create_states else False
        obs, reward, terminated, truncated, info = super().step(action)
        
        if self.create_states:
            prev_len = len(self.saved_slots)
            self._make_state_sample()
            curr_len = len(self.saved_slots)
            if event_state:
                terminated = truncated = True

        return obs, reward, terminated, truncated, info
