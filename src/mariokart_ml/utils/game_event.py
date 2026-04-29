from abc import ABC, abstractmethod
from typing import cast

import gymnasium as gym
import numpy as np
from desmume.emulator_mkds import MarioKart

from mariokart_ml.utils.collision import compute_collision_dists


class Event(ABC):
    def __call__(self, env: gym.Env) -> bool:
        return self.update(env)

    def __or__(self, other: "Event") -> "Event":
        return AnyEvent([self, other])

    def __and__(self, other: "Event") -> "Event":
        return AllEvent([self, other])

    @abstractmethod
    def update(self, env: gym.Env) -> bool: ...


class AnyEvent(Event):
    def __init__(self, events: list[Event]):
        super().__init__()
        self.events = events

    def update(self, env: gym.Env) -> bool:
        return any(event.update(env) for event in self.events)


class AllEvent(Event):
    def __init__(self, events: list[Event]):
        super().__init__()
        self.events = events

    def update(self, env: gym.Env) -> bool:
        return all(event.update(env) for event in self.events)


class BlankEvent(Event):
    def __init__(self):
        super().__init__()

    def update(self, env: gym.Env) -> bool:
        return True


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
        race_started = env.get_wrapper_attr("race_started")
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
        emu = cast(MarioKart, env.get_wrapper_attr("emu"))
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

        emu = cast(MarioKart, env.get_wrapper_attr("emu"))
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

        emu = cast(MarioKart, env.get_wrapper_attr("emu"))
        status = emu.memory.race_status.driverStatus[0]
        lap_progress = float(status.lapProgress)
        race_progress = float(status.raceProgress)

        lap_end = lap_progress > self.max_progress and race_progress > 0.1
        return lap_end


class CollisionEvent(Event):
    def __init__(self):
        super().__init__()
        self.col_count = 0
        self.race_start = RaceStartEvent()

    def update(self, env: gym.Env) -> bool:
        started = self.race_start.update(env)
        if not started:
            return False

        emu = cast(MarioKart, env.get_wrapper_attr("emu"))
        collision = compute_collision_dists(emu, n_rays=3)

        if collision is None:
            return False

        return bool(np.any(collision < 30.0))


class SlowSpeedEvent(Event):
    def __init__(self):
        super().__init__()
        self.race_start = RaceStartEvent()

    def update(self, env: gym.Env) -> bool:
        started = self.race_start.update(env)
        if not started:
            return False

        emu = cast(MarioKart, env.get_wrapper_attr("emu"))
        speed = float(emu.memory.driver.speed) / max(float(emu.memory.driver.maxSpeed), 1e-8)

        return speed < 0.4
