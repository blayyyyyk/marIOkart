from abc import ABC, abstractmethod
from typing import Callable, Literal, Optional, cast

import gymnasium as gym
from desmume.emulator_mkds import MarioKart

from ..utils.game_event import Event


class MovieWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, path: str, disable_event: Optional[Event] = None):
        super(MovieWrapper, self).__init__(env)
        assert self.has_wrapper_attr(
            "emu"
        ), "Provided environment does not have an emulator attribute. It is recommended to use the MarioKartEnv as your base environment."
        self.movie_path = path

        self.disable_event = disable_event

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
        if self.disable_event is not None and self.disable_event.update(self):
            self._stop_movie()

        info |= self._get_info()
        return obs, reward, terminated, truncated, info

    def close(self):
        emu: MarioKart = self.get_wrapper_attr('emu')
        if emu.movie.is_playing():
            emu.movie.stop()

        super().close()
