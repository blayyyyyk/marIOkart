from abc import ABC, abstractmethod
from typing import Literal, cast

import gymnasium as gym
from desmume.emulator_mkds import MarioKart


class Event(ABC):
    def __init__(self):
        ...

    def __call__(self, env: gym.Env) -> bool:
        return self.update(env)

    @abstractmethod
    def update(self, env: gym.Env) -> bool:
        ...


class BlankEvent(Event):
    def __init__(self):
        super().__init__()

    def update(self, env: gym.Env) -> bool:
        return False


class StepCountEvent(Event):
    def __init__(self, max_steps: int):
        super().__init__()
        self.max_steps = max_steps
        self.step_count = 0

    def update(self, env: gym.Env) -> bool:
        self.step_count += 1
        return self.step_count >= self.max_steps

class RaceStartEvent(Event):
    def __init__(self):
        super().__init__()

    def update(self, env: gym.Env) -> bool:
        race_started = env.get_wrapper_attr('race_started')
        return race_started

class RaceStartOffsetEvent(Event):
    def __init__(self, offset: int):
        super().__init__()
        self.step_count_rule = StepCountEvent(offset)
        self.race_start_rule = RaceStartEvent()

    def update(self, env: gym.Env) -> bool:
        out = self.race_start_rule.update(env)
        if not out:
            return self.step_count_rule.update(env)

        return out


class MovieEndEvent(Event):
    def __init__(self):
        super().__init__()

    def update(self, env: gym.Env) -> bool:
        emu = cast(MarioKart, env.get_wrapper_attr('emu'))
        return not emu.movie.is_playing()


class RaceEndEvent(Event):
    def __init__(self, epsilon=1e-2):
        super().__init__()
        self.max_progress = 1.0 - epsilon
        self.race_start_rule = RaceStartEvent()

    def update(self, env: gym.Env) -> bool:
        started = self.race_start_rule.update(env)
        if not started:
            return False

        emu = cast(MarioKart, env.get_wrapper_attr('emu'))
        progress = float(emu.memory.race_status.driverStatus[0].raceProgress)
        return progress > self.max_progress


class LapEndEvent(Event):
    def __init__(self, epsilon=1e-2):
        super().__init__()
        self.max_progress = 1.0 - epsilon
        self.race_start_rule = RaceStartEvent()

    def update(self, env: gym.Env) -> bool:
        started = self.race_start_rule.update(env)
        if not started:
            return False

        emu = cast(MarioKart, env.get_wrapper_attr('emu'))
        status = emu.memory.race_status.driverStatus[0]
        lap_progress = float(status.lapProgress)
        race_progress = float(status.raceProgress)

        lap_end = lap_progress > self.max_progress and race_progress > 0.1
        return lap_end

RuleMapLiteral = Literal["race_start", "movie_end", "race_end", "lap_end"]
RULE_MAP: dict[str, type[Event]] = {
    "race_start": RaceStartEvent,
    "movie_end": MovieEndEvent,
    "race_end": RaceEndEvent,
    "lap_end": LapEndEvent
}
