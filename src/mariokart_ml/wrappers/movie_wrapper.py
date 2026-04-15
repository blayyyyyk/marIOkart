import gymnasium as gym
from desmume.emulator_mkds import MarioKart
from typing import Optional, Callable, Literal, cast
from abc import ABC, abstractmethod

class UpdateRule(ABC):
    def __init__(self):
        ...
       
    def __call__(self, env: gym.Env) -> bool:
        return self.update(env)
        
    @abstractmethod
    def update(self, env: gym.Env) -> bool:
        ...

class MovieWrapper(gym.Wrapper):
    movie_update_rule: UpdateRule
    
    def __init__(self, env: gym.Env, path: str, max_steps: int | Literal["race_start", "movie_end", "race_end", "lap_end"] = "movie_end"):
        super(MovieWrapper, self).__init__(env)
        assert self.has_wrapper_attr(
            "emu"
        ), "Provided environment does not have an emulator attribute. It is recommended to use the MarioKartEnv as your base environment."
        self.movie_path = path
        
        funcs: dict[str, type[UpdateRule]] = {
            "race_start": RaceStartUpdateRule,
            "movie_end": MovieEndUpdateRule,
            "race_end": RaceEndUpdateRule,
            "lap_end": LapEndUpdateRule
        }
        if isinstance(max_steps, str):
            self.movie_update_rule = funcs[max_steps]()
        elif isinstance(max_steps, int):
            self.movie_update_rule = RaceStartOffsetUpdateRule(max_steps)
        else:
            raise ValueError(f"Invalid max_steps: {max_steps}")
                
        self.movie_played = False

    def reset(self, *, seed=None, options=None):
        out = super().reset()
        if not self.movie_played:
            emu: MarioKart = self.get_wrapper_attr('emu')
            emu.movie.play(self.movie_path)
            self.movie_played = True
        
        return out

    def _get_info(self):
        emu: MarioKart = self.get_wrapper_attr('emu')
        return {
            "movie_playing": emu.movie.is_playing()
        }

    def _stop_movie(self):
        emu: MarioKart = self.get_wrapper_attr('emu')
        if emu.movie.is_playing():
            emu.movie.stop()

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self.movie_update_rule is not None:
            if not self.movie_update_rule(self):
                self._stop_movie()

        info |= self._get_info()
        return obs, reward, terminated, truncated, info

    def close(self):
        emu: MarioKart = self.get_wrapper_attr('emu')
        if emu.movie.is_playing():
            emu.movie.stop()

        super().close()



class StepCountUpdateRule(UpdateRule):
    def __init__(self, max_steps: int):
        super().__init__()
        self.max_steps = max_steps
        self.step_count = 0

    def update(self, env: gym.Env) -> bool:
        self.step_count += 1
        return self.step_count >= self.max_steps

class RaceStartUpdateRule(UpdateRule):
    def __init__(self):
        super().__init__()

    def update(self, env: gym.Env) -> bool:
        emu = env.get_wrapper_attr('emu')
        race_started = env.get_wrapper_attr('race_started')
        return not race_started
        
class RaceStartOffsetUpdateRule(UpdateRule):
    def __init__(self, offset: int):
        super().__init__()
        self.step_count_rule = StepCountUpdateRule(offset)
        self.race_start_rule = RaceStartUpdateRule()
        
    def update(self, env: gym.Env) -> bool:
        out = self.race_start_rule.update(env)
        if not out:
            return self.step_count_rule.update(env)
        
        return out
    

class MovieEndUpdateRule(UpdateRule):
    def __init__(self):
        super().__init__()
        
    def update(self, env: gym.Env) -> bool:
        emu = cast(MarioKart, env.get_wrapper_attr('emu'))
        return emu.movie.is_playing()
        

class RaceEndUpdateRule(UpdateRule):
    def __init__(self, epsilon=1e-2):
        super().__init__()
        self.max_progress = 1.0 - epsilon
        self.race_start_rule = RaceStartUpdateRule()
        self.movie_end_rule = MovieEndUpdateRule()
        
    def update(self, env: gym.Env) -> bool:
        started = self.race_start_rule.update(env)
        if started:
            return True
        
        emu = cast(MarioKart, env.get_wrapper_attr('emu'))
        progress = float(emu.memory.race_status.driverStatus[0].raceProgress)
        return progress <= self.max_progress and self.movie_end_rule.update(env)
        
        
class LapEndUpdateRule(UpdateRule):
    def __init__(self, epsilon=1e-2):
        super().__init__()
        self.max_progress = 1.0 - epsilon
        self.race_start_rule = RaceStartUpdateRule()
        self.movie_end_rule = MovieEndUpdateRule()
        
    def update(self, env: gym.Env) -> bool:
        started = self.race_start_rule.update(env)
        if started:
            return True
        
        emu = cast(MarioKart, env.get_wrapper_attr('emu'))
        status = emu.memory.race_status.driverStatus[0]
        lap_progress = float(status.lapProgress)
        race_progress = float(status.raceProgress)
        
        movie_end = self.movie_end_rule.update(env)
        lap_end = lap_progress <= self.max_progress or race_progress <= 0.1
        return lap_end and movie_end
        

