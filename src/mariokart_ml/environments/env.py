import math
import time
from contextlib import nullcontext
from functools import cached_property
from typing import Any, Literal, TypedDict, cast

import gymnasium as gym
import numpy as np
from desmume.emulator import SCREEN_HEIGHT, SCREEN_PIXEL_SIZE, SCREEN_WIDTH
from desmume.emulator_mkds import MarioKart, get_fx
from gymnasium.wrappers.utils import RunningMeanStd

from mariokart_ml.config import N_KEYS, RAY_MAX_DIST
from mariokart_ml.utils.checkpoint import NKM
from mariokart_ml.utils.collision import compute_collision_dists
from mariokart_ml.utils.game_event import Event, RaceEndEvent, SlowSpeedEvent
from mariokart_ml.utils.suppress import Suppress
from mariokart_ml.wrappers.boundary_wrapper import project_2d
from mariokart_ml.wrappers.checkpoint_wrapper import checkpoint_angle_signed

ROTATION_CONST = 1 / (1 << 15)

GAME_SAVE_SLOT = 0  # will bring the kart back to the start of the game menu
RACE_SAVE_SLOT = 1  # will bring the kart back to the start of the race
FRAME_SAVE_SLOT = 2  # will bring the kart back to a checkpoint in the race


def fmt_obs(space: gym.spaces.Space, obs: float | np.ndarray) -> np.ndarray:
    if isinstance(obs, np.ndarray):
        return obs
    return np.array(obs, space.dtype).reshape(space.shape)


def fmt_space_dict(space: gym.spaces.Dict, obs: dict[str, float]) -> dict[str, np.ndarray]:
    assert set(space.spaces.keys()) == set(obs.keys()), f"Expected keys {set(space.spaces.keys())}, got {set(obs.keys())}"
    return {k: fmt_obs(s, obs[k]) for k, s in space.items()}


class ResetOptions(TypedDict):
    reset_type: Literal["respawn", "savestate", "menu", "custom"] | int


class TimeTrialEnv(gym.Env[dict[str, Any], int]):
    emu: MarioKart

    def __init__(
        self,
        rom_path: str,
        reset_event: Event | None = None,
        suppress_desmume: bool = True,
    ):
        super().__init__()
        if reset_event is None:
            reset_event = RaceEndEvent() | SlowSpeedEvent()

        self.observation_space: gym.spaces.Dict = gym.spaces.Dict(
            {
                "pos_x": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                "pos_y": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                "pos_z": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                "vel_x": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                "vel_y": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                "vel_z": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                "surf_norm_x": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                "surf_norm_y": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                "surf_norm_z": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            }
        )

        self.action_space = gym.spaces.Discrete(2**N_KEYS, dtype=np.uint16)

        self.metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

        self.render_mode = "rgb_array"
        self.suppress_desmume = suppress_desmume

        with Suppress() if self.suppress_desmume else nullcontext():
            # stores emulator instance
            self.emu = MarioKart()
            self.emu.open(rom_path)
            self.emu.volume_set(0)

            # clear out any leftover save slots, replace with main menu
            self.emu.savestate.save(RACE_SAVE_SLOT)
            self.emu.savestate.save(FRAME_SAVE_SLOT)

        self.reset_event = reset_event

        # stores position of kart at time when race starts
        self.race_started = False
        self._starting_position = np.array([2000, -2000, 2000], dtype=np.float32)

        self.rms_latency = RunningMeanStd()

    def _get_obs(self) -> dict:
        assert isinstance(self.observation_space, gym.spaces.Dict)
        if not self.race_started:
            return {k: 0.0 for k in self.observation_space.keys()}

        observation = {}

        # kart position
        position = self.emu.memory.driver_position
        position_norm = position / np.linalg.norm(position)
        observation["pos_x"] = position_norm[0]
        observation["pos_y"] = position_norm[1]
        observation["pos_z"] = position_norm[2]

        # kart velocity
        velocity = self.emu.memory.driver_velocity
        velocity_norm = velocity / np.linalg.norm(velocity)
        observation["vel_x"] = velocity_norm[0]
        observation["vel_y"] = velocity_norm[1]
        observation["vel_z"] = velocity_norm[2]

        # kart up vector / surface normal vector
        surface_normal = get_fx(self.emu.memory.driver.upDir, shape=(3,))
        observation["surf_norm_x"] = surface_normal[0]
        observation["surf_norm_y"] = surface_normal[1]
        observation["surf_norm_z"] = surface_normal[2]

        return fmt_space_dict(self.observation_space, observation)

    def _get_info(self):
        if self.race_started:
            race_progress = float(self.emu.memory.race_status.driverStatus[0].raceProgress)
            lap_progress = float(self.emu.memory.race_status.driverStatus[0].lapProgress)
        else:
            race_progress = 0.0
            lap_progress = 0.0

        return {
            "race_started": self._race_active(),
            "race_progress": race_progress,
            "lap_progress": lap_progress,
        }

    def _race_active(self):
        _f = 0
        try:
            _f = self.emu.memory.race_state.frameCounter - self.emu.memory.race_state.frameCounter2
        except Exception as _e:
            return False

        return _f == 1

    def step(self, action):
        assert action.dtype == np.uint16, "action must be a numpy array of uint16"

        start = time.time()
        self.emu.input.keypad_update(0)
        if not self.emu.movie.is_playing():
            # this might be redundant but make sure keys are cleared
            self.emu.input.keypad_update(int(action))

        # update emulator state
        self.emu.cycle()
        if self._race_active() and not self.race_started:
            self.race_started = True
            self.emu.savestate.save(RACE_SAVE_SLOT)
            self.emu.savestate.save(FRAME_SAVE_SLOT)
        elif not self._race_active() and self.race_started:
            self.race_started = False

        obs = self._get_obs()
        info = self._get_info()

        terminated = self.reset_event.update(self) if self.reset_event is not None else False
        truncated = terminated
        end = time.time()

        self.rms_latency.update(np.array([end - start]))
        info["latency"] = float(self.rms_latency.mean)

        reward = 0.0

        return obs, reward, terminated, truncated, info

    def render(self):
        mem = self.emu.display_buffer_as_rgbx()
        top = mem[: SCREEN_PIXEL_SIZE * 4]
        bottom = mem[SCREEN_PIXEL_SIZE * 4 :]

        arr_t = np.ndarray(shape=(SCREEN_HEIGHT, SCREEN_WIDTH, 4), dtype=np.uint8, buffer=top)
        arr_b = np.ndarray(shape=(SCREEN_HEIGHT, SCREEN_WIDTH, 4), dtype=np.uint8, buffer=bottom)

        arr = np.concatenate([arr_t, arr_b], axis=0)

        return arr[:, :, :3]

    def reset(self, *, seed=None, options=None):
        """Gymnasium requires a reset method to restart the environment."""
        super().reset(seed=seed)
        if options is None:
            options = {"reset_type": None}

        obs = self._get_obs()
        info = self._get_info()

        options = {} if options is None else options
        reset_type = options.get("reset_type", None)

        with Suppress() if self.suppress_desmume else nullcontext():
            if isinstance(reset_type, int):
                self.emu.savestate.load(reset_type)

            if self.emu.movie.is_playing():
                self.emu.movie.stop()

        self.emu.volume_set(0)

        return obs, info

    def close(self):
        self.emu.close()
        super().close()


class TimeTrialObservations(gym.ObservationWrapper):
    def __init__(self, env: gym.Env[gym.spaces.Dict, gym.spaces.Discrete]):
        super().__init__(env)

        assert isinstance(env.observation_space, gym.spaces.Dict)
        self.observation_space = gym.spaces.Dict(
            {
                "speed_frac": gym.spaces.Box(low=0, high=1.3, shape=(1,), dtype=np.float32),
                "speed_max_frac": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "speed_rescaled": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                # "speed_turn": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                "speed_offroad": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                "speed_effect": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                "speed_midair": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                "angle_checkpoint_near": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                "angle_checkpoint_mid": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                "angle_checkpoint_far": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                "angle_curvature_near": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                "angle_curvature_mid": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                "angle_curvature_far": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                "angle_facing": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                "angle_drift": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                "angle_inertia": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                "angle_pitch": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                "drift_direction": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                "drift_progress": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "drift_left_count": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "drift_right_count": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "drift_boost_active": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "drift_timeout": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "centerline_dist": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                "surf_friction": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "is_touching_ground": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "trackstatus": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "prb_flag": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "collision_dist_left": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                "collision_dist_right": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                "collision_dist_front": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                "collision_is_touching": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                # "horizon_short":    gym.spaces.Box(low=-1,   high=1,   shape=(1,), dtype=np.float32),
                # "horizon_mid":      gym.spaces.Box(low=-1,   high=1,   shape=(1,), dtype=np.float32),
                # "horizon_long":     gym.spaces.Box(low=-1,   high=1,   shape=(1,), dtype=np.float32),
                # "obsppm" :             gym.spaces.Box(low=-1,   high=1,   shape=(1,), dtype=np.float32),
            }
            | env.observation_space.spaces
        )

        self.rms_centerline_dist = RunningMeanStd()
        self.rms_speed_turn = RunningMeanStd()
        self.rms_collision_dist = RunningMeanStd(shape=(3,))

    def _centerline_dist(self):
        emu: MarioKart = cast(MarioKart, self.get_wrapper_attr("emu"))
        if not emu.memory.race_ready:
            return 0.0

        p0, p1 = np.unstack(emu.memory.checkpoint_pos()["current_checkpoint_pos"], axis=0)
        mid0 = (p0 + p1) / 2

        p2, p3 = np.unstack(emu.memory.checkpoint_pos()["next_checkpoint_pos"], axis=0)
        mid1 = (p2 + p3) / 2

        kart_position = emu.memory.driver_position
        centerline_intersect = project_2d(mid0[None, :], mid1[None, :], kart_position)
        centerline_dist = np.linalg.norm(centerline_intersect - kart_position)

        centerline_dist_old = centerline_dist
        norm = (centerline_dist - self.rms_centerline_dist.mean) / (self.rms_centerline_dist.var ** (0.5) + 1e-8)  # noqa: F841

        self.rms_centerline_dist.update(np.array([centerline_dist_old]))

        return centerline_dist

    def _speed_turn(self, emu: MarioKart):
        speed_turn = float(emu.memory.driver.yRotSpeed)
        speed_turn_old = speed_turn
        self.rms_speed_turn.update(np.array([speed_turn]))
        return (speed_turn_old - self.rms_speed_turn.mean) / (self.rms_speed_turn.var ** (0.5) + 1e-8)

    def _get_speed_obs(self, emu: MarioKart):
        observation = {}

        # kart speed
        eps = 1e-8
        speed_diff = float(emu.memory.driver.maxSpeed) - float(emu.memory.driver.speed)
        observation["speed_frac"] = float(emu.memory.driver.speed) / (float(emu.memory.driver.maxSpeed) + eps)
        observation["speed_max_frac"] = float(emu.memory.driver.maxSpeedFraction)
        observation["speed_rescaled"] = speed_diff / (float(emu.memory.driver.maxSpeed) + eps)  # used to be: speed_deficit
        # observation["speed_turn"] = emu.memory.driver.yRotSpeed * FX32_SCALE_FACTOR

        # special speed
        observation["speed_offroad"] = float(emu.memory.driver.speedMultiplier)
        observation["speed_effect"] = float(emu.memory.driver.field394)  # this is a discovered field!
        observation["speed_midair"] = float(emu.memory.driver.field3F8)  # this is a discovered field! # was: air_speed

        return observation

    def _get_collision_obs(self, emu: MarioKart):
        observation = {}

        dists = compute_collision_dists(emu, n_rays=3)

        if dists is None:
            dists = np.zeros(3)

        observation["collision_dist_left"] = dists[0] / RAY_MAX_DIST
        observation["collision_dist_right"] = dists[1] / RAY_MAX_DIST
        observation["collision_dist_front"] = dists[2] / RAY_MAX_DIST
        observation["collision_is_touching"] = 1.0 if np.any(dists < 15.0) else 0.0

        return observation

    def _get_angle_obs(self, emu: MarioKart):
        observation = {}

        observation["angle_facing"] = emu.memory.driver.yRot * ROTATION_CONST
        observation["angle_drift"] = emu.memory.driver.driftRotY * ROTATION_CONST
        observation["angle_pitch"] = emu.memory.driver.xRot * ROTATION_CONST

        def wrap_angle_diff(diff: float) -> float:
            # Keeps the difference strictly between -1 and 1
            return (diff + 1.0) % 2.0 - 1.0

        a1 = checkpoint_angle_signed(emu, direction_mode="movement")
        a2 = checkpoint_angle_signed(emu, direction_mode="movement", lookahead_id=2)
        a3 = checkpoint_angle_signed(emu, direction_mode="movement", lookahead_id=3)

        observation["angle_checkpoint_near"] = a1
        observation["angle_checkpoint_mid"] = wrap_angle_diff(a2 - a1)
        observation["angle_checkpoint_far"] = wrap_angle_diff(a3 - a2)

        b1 = self.nkm.get_cpoi_angle_interp(mid_offset=0.02) / np.pi
        b2 = self.nkm.get_cpoi_angle_interp(mid_offset=0.04) / np.pi
        b3 = self.nkm.get_cpoi_angle_interp(mid_offset=0.06) / np.pi

        observation["angle_curvature_near"] = b1
        observation["angle_curvature_mid"] = b2
        observation["angle_curvature_far"] = b3
        observation["angle_inertia"] = np.arctan2(emu.memory.driver_direction, emu.memory.driver_velocity)[0].item() / math.pi  # was: heading_rate

        return observation

    def _get_drift_obs(self, emu: MarioKart):
        observation = {}

        observation["drift_left_count"] = emu.memory.driver.driftLeftCount  # was: mt_left
        observation["drift_right_count"] = emu.memory.driver.driftRightCount  # was: mt_right
        observation["drift_timeout"] = emu.memory.driver.driftLeftRightTimeout  # was: mttime
        observation["drift_direction"] = float(emu.memory.driver.leftRightDir)
        observation["drift_boost_active"] = 1.0 if emu.memory.driver.driftBoostCounter > 0 else -1.0  # was: slip_angle
        observation["drift_progress"] = (observation["drift_left_count"] + observation["drift_right_count"]) / 4

        return observation

    def _get_track_obs(self, emu: MarioKart):
        observation = {}
        # out["horizon_short"] = 0.0 # TODO
        # out["horizon_mid"] = 0.0 # TODO
        # out["horizon_long"] = 0.0 # TODO
        # out["obsppm"] = 0.0 # TODO
        return observation

    def _get_special_obs(self, emu: MarioKart):
        observation = {}

        observation["is_touching_ground"] = 0.0 if emu.memory.driver.floorColType > 0 else 1.0
        observation["centerline_dist"] = self._centerline_dist()
        observation["surf_friction"] = float(emu.memory.driver.velocityMinusDirMultiplier)
        observation["trackstatus"] = 0.0  # TODO: Ask symbiose
        observation["prb_flag"] = int((emu.memory.driver.flags & 0x20) != 0)  # pre-respawn bit (prb)

        return observation

    @cached_property
    def nkm(self) -> NKM:
        emu = cast(MarioKart, self.get_wrapper_attr("emu"))
        nkm = NKM(emu)
        return nkm

    def observation(self, observation: dict):
        emu: MarioKart = cast(MarioKart, self.get_wrapper_attr("emu"))
        assert isinstance(self.observation_space, gym.spaces.Dict), "wrong observation space type"
        if not emu.memory.race_ready:
            return {k: fmt_obs(s, 0.0) for k, s in self.observation_space.items()}

        observation |= self._get_speed_obs(emu)
        observation |= self._get_angle_obs(emu)
        observation |= self._get_drift_obs(emu)
        observation |= self._get_track_obs(emu)
        observation |= self._get_collision_obs(emu)
        observation |= self._get_special_obs(emu)

        return fmt_space_dict(self.observation_space, observation)
