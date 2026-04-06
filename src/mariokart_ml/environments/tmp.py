# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 20:36:20 2024
@author: kenzl
"""
import sys
import os
import gym
from gym import spaces
import numpy
from collections import deque
from desmume.emulator import DeSmuME
import math
import random
import pandas as pd

# This whole class is for creating mini races
# for uniform training over the track
class CenterLineTracker:
    def __init__(self, filepath: str, target_points: int = 20000):
        df = pd.read_csv(filepath, sep=r"\s+|\t+", engine="python")
        required = ["xpos", "ypos", "zpos"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in center_line.txt: {missing}")

        self.x = df["xpos"].to_numpy(dtype=numpy.float64)
        self.y = df["ypos"].to_numpy(dtype=numpy.float64)
        self.z = df["zpos"].to_numpy(dtype=numpy.float64)
        self.sin = df["sine"].to_numpy(dtype=numpy.float64)
        self.cos = df["cosi"].to_numpy(dtype=numpy.float64)

        if len(self.x) < 2:
            raise ValueError("center_line.txt must contain at least 2 points")

        diff_x = numpy.diff(self.x)
        diff_y = numpy.diff(self.y)
        diff_z = numpy.diff(self.z)
        segment_lengths = numpy.sqrt(diff_x**2 + diff_y**2 + diff_z**2)
        total_distance = numpy.sum(segment_lengths)
        target_distances = numpy.linspace(0, total_distance, target_points)
        cumulative_distances = numpy.cumsum(numpy.insert(segment_lengths, 0, 0))

        self.new_x = self.linear_interpolate(cumulative_distances, self.x, target_distances)
        self.new_y = self.linear_interpolate(cumulative_distances, self.y, target_distances)
        self.new_z = self.linear_interpolate(cumulative_distances, self.z, target_distances)
        self.new_sin = self.linear_interpolate(cumulative_distances, self.sin, target_distances)
        self.new_cos = self.linear_interpolate(cumulative_distances, self.cos, target_distances)

        self.points = numpy.column_stack((self.new_x, self.new_y, self.new_z))
        self.seg_start = self.points[:-1]
        self.seg_end = self.points[1:]
        self.seg_vec = self.seg_end - self.seg_start
        self.seg_len_sq = numpy.sum(self.seg_vec ** 2, axis=1) + 1e-12

        norm = numpy.sqrt(self.new_sin**2 + self.new_cos**2) + 1e-12
        self.new_sin /= norm
        self.new_cos /= norm
        self.new_angle = numpy.arctan2(self.new_sin, self.new_cos)

    def linear_interpolate(self, cum_distances, values, target_distances):
        idx = numpy.searchsorted(cum_distances, target_distances) - 1
        idx = numpy.clip(idx, 0, len(cum_distances) - 2)
        x0 = cum_distances[idx]
        x1 = cum_distances[idx + 1]
        y0 = values[idx]
        y1 = values[idx + 1]
        slope = (y1 - y0) / (x1 - x0)
        return y0 + slope * (target_distances - x0)

    def signed_distance_to_centerline(self, xpos: float, ypos: float, zpos: float):
        p = numpy.array([xpos, ypos, zpos], dtype=numpy.float64)
        w = p - self.seg_start
        t = numpy.sum(w * self.seg_vec, axis=1) / self.seg_len_sq
        t = numpy.clip(t, 0.0, 1.0)
        closest = self.seg_start + self.seg_vec * t[:, None]
        diff = p - closest
        dist_sq = numpy.sum(diff ** 2, axis=1)
        best_idx = numpy.argmin(dist_sq)
        a = self.seg_start[best_idx]
        b = self.seg_end[best_idx]
        seg = b - a
        rel = p - a
        cross = seg[0] * rel[1] - seg[1] * rel[0]
        unsigned_dist = numpy.sqrt(dist_sq[best_idx])
        signed_dist = unsigned_dist if cross > 0 else -unsigned_dist
        signed_dist = signed_dist / 800000
        signed_dist = numpy.clip(signed_dist, -1, 1)
        t_best = float(t[best_idx])
        progress_float = best_idx + t_best
        return float(signed_dist), int(best_idx), progress_float

    def angle_to_centerline(self, player_sin, player_cos, current_cp_sin, current_cp_cos):
        cross = player_cos * current_cp_sin - player_sin * current_cp_cos
        return cross
    
    def min_max_centerline(self): # retrieve min max x,y,z positions to normalize xyz pos
        min_x = min(self.x)
        max_x = max(self.x)
        min_y = min(self.y)
        max_y = max(self.y)
        min_z = min(self.z)
        max_z = max(self.z)
        
        return min_x, max_x, min_y, max_y, min_z, max_z


class EmulatorEnvNoXY_v12(gym.Env):

    def __init__(self, metrics, moves, angle_vals, utils, isrender='human', fixedstart='false',
                 early_term='false', episodelength=960):
        super(EmulatorEnvNoXY_v12, self).__init__()

        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        self.emu = DeSmuME()
        self.emu.set_num_cores(0)
        self.emu.set_sound_enabled(False)
        
        if isrender=='human':
            self.emu.set_3d_renderer(1)
            self.window = self.emu.create_sdl_window()

        else:
            self.emu.set_3d_renderer(0)
            for layer in range(5):
                self.emu.gpu_set_layer_main_enable_state(layer, False)
                self.emu.gpu_set_layer_sub_enable_state(layer, False)
        
        
        self.emu.set_jit_enabled(True, 100) # 100 = fastest block size
        self.emu.open('./mkdsai/mkds_usa.nds')
        sys.stdout = old_stdout

        self.metrics = metrics
        self.isrender = isrender
        self.angle_vals = angle_vals
        self.moves = moves
        self.early_term = early_term
        self.utils = utils
        self.fixedstart = fixedstart
        self.episodelength = episodelength


        self.frame_buffer_size = 5
        self.speed_queue = deque([0] * self.frame_buffer_size, maxlen=self.frame_buffer_size)
        self.mt_queue = deque([0] * 2, maxlen=2)
        self.frame_buffer = deque(maxlen=self.frame_buffer_size)

        self.savestate_number = 0
        self.past_trackstatus_by_savestate = {}

        self.action_space = spaces.Discrete(5)

        # ------------------------------------------------------------------ #
        # OBSERVATION SPACE
        # ------------------------------------------------------------------ #
        self.observation_space = spaces.Dict({
            # --- Existing ---
            "speed":            spaces.Box(low=0,    high=1.3, shape=(1,), dtype=numpy.float32),
            "angle":            spaces.Box(low=-1,   high=1,   shape=(1,), dtype=numpy.float32),
            "angle_facing":     spaces.Box(low=-1,   high=1,   shape=(1,), dtype=numpy.float32),
            "drift_angle":      spaces.Box(low=-1,   high=1,   shape=(1,), dtype=numpy.float32),
            "is_collision":     spaces.Box(low=0,    high=1,   shape=(1,), dtype=numpy.float32),
            "centerline_dist":  spaces.Box(low=-1,   high=1,   shape=(1,), dtype=numpy.float32),
            "grip":             spaces.Box(low=0,    high=1,   shape=(1,), dtype=numpy.float32),
            "drift_direction":  spaces.Box(low=-1,   high=1,   shape=(1,), dtype=numpy.float32),
            "drift_progress":   spaces.Box(low=0,    high=1,   shape=(1,), dtype=numpy.float32),
            "mt_left":          spaces.Box(low=0,    high=1,   shape=(1,), dtype=numpy.float32),
            "mt_right":         spaces.Box(low=0,    high=1,   shape=(1,), dtype=numpy.float32),
            "mttime":           spaces.Box(low=0,    high=1,   shape=(1,), dtype=numpy.float32),
            "air":              spaces.Box(low=0,    high=1,   shape=(1,), dtype=numpy.float32),
            "trackstatus":      spaces.Box(low=0,    high=1,   shape=(1,), dtype=numpy.float32),
            
            "xpos_norm":  spaces.Box(low=-1,   high=1,   shape=(1,), dtype=numpy.float32),
            "ypos_norm":  spaces.Box(low=-1,   high=1,   shape=(1,), dtype=numpy.float32),
            "zpos_norm":  spaces.Box(low=-1,   high=1,   shape=(1,), dtype=numpy.float32),

            # --- NEW: velocity vector ---
            # Actual per-frame displacement in kart-local coords, fixed-point /4096.
            "vel_x":             spaces.Box(low=-1, high=1,  shape=(1,), dtype=numpy.float32),
            "vel_y":             spaces.Box(low=-1, high=1,  shape=(1,), dtype=numpy.float32),
            "vel_z":             spaces.Box(low=-1, high=1,  shape=(1,), dtype=numpy.float32),
            "vertical_velocity": spaces.Box(low=-1, high=1,  shape=(1,), dtype=numpy.float32),

            # Slip angle: facing direction vs actual movement direction, [-1,1]
            "slip_angle":       spaces.Box(low=-1,   high=1,   shape=(1,), dtype=numpy.float32),

            # Nose up/down angle, same normalisation as angle_facing.
            "pitch":            spaces.Box(low=-1,   high=1,   shape=(1,), dtype=numpy.float32),
            # Surface normal XYZ: encodes road slope and banking.
            "surf_norm_x":      spaces.Box(low=-1,   high=1,   shape=(1,), dtype=numpy.float32),
            "surf_norm_y":      spaces.Box(low=-1,   high=1,   shape=(1,), dtype=numpy.float32),
            "surf_norm_z":      spaces.Box(low=-1,   high=1,   shape=(1,), dtype=numpy.float32),
            # --- NEW: speed modifiers ---
            # offroad < 1 means grass/sand penalty; effect > 1 means item boost.
            "offroad_speed":     spaces.Box(low=-1, high=1,  shape=(1,), dtype=numpy.float32),

            "effect_speed":      spaces.Box(low=-1, high=1,  shape=(1,), dtype=numpy.float32),
            "air_speed":         spaces.Box(low=-1, high=1,  shape=(1,), dtype=numpy.float32),

            # Binary: 1 = MT boost currently firing.
            "boost_mt":         spaces.Box(low=0,    high=1,   shape=(1,), dtype=numpy.float32),
            
            # Fraction of max speed currently allowed by engine [0, ~1.3].
            "max_speed_fraction":spaces.Box(low=0,  high=1,  shape=(1,), dtype=numpy.float32),

            # --- NEW: respawn ---
            "prb_flag":         spaces.Box(low=0,    high=1,   shape=(1,), dtype=numpy.float32),
            # --- NEW: derived ---
            "heading_rate":     spaces.Box(low=-1,   high=1,   shape=(1,), dtype=numpy.float32),
            "speed_deficit":    spaces.Box(low=0,    high=1,   shape=(1,), dtype=numpy.float32),
            # --- NEW: track horizon / curvature ---
            "horizon_short":    spaces.Box(low=-1,   high=1,   shape=(1,), dtype=numpy.float32),
            "horizon_mid":      spaces.Box(low=-1,   high=1,   shape=(1,), dtype=numpy.float32),
            "horizon_long":     spaces.Box(low=-1,   high=1,   shape=(1,), dtype=numpy.float32),
            "obsppm" :             spaces.Box(low=-1,   high=1,   shape=(1,), dtype=numpy.float32),

        })

        # ------------------------------------------------------------------ #
        # INSTANCE VARIABLES — existing
        # ------------------------------------------------------------------ #
        self.checkpoints = 0
        self.lap_f = 0
        self.max_speed = 0
        self.speed = 0
        self.xpos = 0
        self.ypos = 0
        self.zpos = 0
        self.angle_facing = 0
        self.drift_angle = 0
        self.sum_angle = 0
        self.current_cp_angle = 0
        self.grip = 0
        self.turn_loss = 0
        self.air = 0
        self.mttime = 0
        self.mt_left = 0
        self.mt_right = 0
        self.drift_direction = 0
        self.drift_progress = 0
        self.boost_timer = 0
        self.angle = 0
        self.real_speed = 0
        self.trackstatus = 0
        self.total_distance = 0
        self.ppm = 0
        self.positions = deque([(0, 0, 0)] * 2, maxlen=2)
        self.position = (0, 0, 0)
        self.current_cp_sin = 0
        self.current_cp_cos = 0

        # NEW instance variables
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.vel_z = 0.0
        self.vertical_velocity = 0.0
        self.slip_angle = 0.0
        self.frames_in_air = 0.0
        self.pitch = 0.0
        self.surf_norm_x = 0.0
        self.surf_norm_y = 1.0      # default: flat ground
        self.surf_norm_z = 0.0
        self.offroad_speed = 1.0
        self.effect_speed = 1.0
        self.air_speed = 1.0
        self.boost_mt = 0.0
        self.max_speed_fraction = 1.0
        self.prb_flag = 0
        self.heading_rate = 0.0
        self.speed_deficit = 0.0
        self._prev_angle_facing = 0.0   # needed for heading_rate computation
        
        self.heading_rate = 0.0
        self.speed_deficit = 0.0
        self._prev_angle_facing = 0.0
        
        # Horizon instance variables
        self.horizon_short = 0.0
        self.horizon_mid = 0.0
        self.horizon_long = 0.0
        self.obsppm = 0.0

        # ------------------------------------------------------------------ #
        # RESET VARS
        # ------------------------------------------------------------------ #
        self.reset_emulator()
        self.frame_count = 0
        self.current_checkpoint = 0
        self.next_checkpoint = 1
        self.total_reward = 0
        self.done = False
        self.speed_queue = deque([0] * 5, maxlen=5)
        self.trackstatus_queue = deque([0] * 2, maxlen=2)
        self.total_distance_queue = deque([0] * 2, maxlen=2)
        self.collision_n = 0
        self.max_kart_speed_default = 0
        self.mt_cooldown = 0
        self.is_collision = 0
        self.wallclip = 1
        
        # mt_state tracking
        self.was_drifting = False
        self.is_drifting = False
        self.just_released_drift = False
        self._prev_drift_direction=0

        # Centerline
        self.centerline = CenterLineTracker(
            "C:/Users/Administrator/Documents/GitHub/mkdsai/mkdsai/center_line/figure_eight_center_v2.txt",
            target_points=20000)

        self.progress_float = 0.0
        self.prev_progress_float = 0.0
        self.progress_frac = 0.0
        self.centerline_dist = 0.0
        self.centerline_seg_idx = 0
        self.centerline_seg_queue = deque([0] * 2, maxlen=2)
        
        self.min_x, self.max_x, self.min_y, self.max_y, self.min_z, self.max_z = self.centerline.min_max_centerline()
        
        print("initialized")

    # ---------------------------------------------------------------------- #
    # INTERNAL HELPERS (unchanged)
    # ---------------------------------------------------------------------- #


    def reset_emulator(self):
        try:
            if self.fixedstart == 'true':
                self.savestate_number = 1
            else:
                self.savestate_number = random.randint(1, 94)
            savestate_file = (
                f"C:/Users/Administrator/Documents/GitHub/mkdsai/mkdsai/center_line/save_slot/figure_height_all.ds{self.savestate_number}"
            )
            self.emu.savestate.load_file(savestate_file)
            print(f"Loaded savestate {self.savestate_number} successfully")
        except Exception as e:
            print(f"Error loading savestate: {e}")
            raise

    # ---------------------------------------------------------------------- #
    # UPDATE STATE — extend with new RAM fields
    # ---------------------------------------------------------------------- #
    def update_state(self, state):
        # Existing fields
        self.checkpoints = state.get("checkpoints")
        self.lap_f = state.get("lap_f")
        self.max_speed = state.get("max_speed")
        self.speed = state.get("speed")
        self.xpos = state.get("xpos")
        self.ypos = state.get("ypos")
        self.zpos = state.get("zpos")
        self.angle_facing = state.get("angle_facing")
        self.drift_angle = state.get("drift_angle")
        self.real_angle = state.get("real_angle")
        self.player_sin = state.get("tSine")
        self.player_cos = state.get("tCosi")
        self.current_cp_angle = state.get("current_cp_angle")
        self.grip = state.get("grip")
        self.turn_loss = state.get("turn_loss")
        self.wallclip = state.get("wallclip")
        self.air = state.get("air")
        self.boost_timer = state.get("boost_timer")
        self.drift_direction = state.get("drift_direction")
        self.drift_progress = state.get("drift_progress")
        self.mttime = state.get("mttime")
        self.mt_left = state.get("mt_left") / 2
        self.mt_right = state.get("mt_right") / 2
        self.trackstatus = state.get("trackstatus")
        self.total_distance = state.get("total_distance")
        self.position = (self.xpos, self.ypos, self.zpos)

        # Centerline (unchanged)
        self.centerline_dist, self.centerline_seg_idx, self.progress_float = \
            self.centerline.signed_distance_to_centerline(self.xpos, self.ypos, self.zpos)
        self.current_cp_sin = self.centerline.new_sin[self.centerline_seg_idx]
        self.current_cp_cos = self.centerline.new_cos[self.centerline_seg_idx]
        self.angle = self.centerline.angle_to_centerline(
            self.player_sin, self.player_cos, self.current_cp_sin, self.current_cp_cos)

        # ------------------------------------------------------------------ #
        # NEW: velocity vector
        # ------------------------------------------------------------------ #
        # NEW: derived — speed deficit
        self.vel_x = state.get("vel_x", 0.0)
        self.vel_y = state.get("vel_y", 0.0)
        self.vel_z = state.get("vel_z", 0.0)
        self.vertical_velocity = state.get("vertical_velocity", 0.0)
        self.slip_angle = state.get("slip_angle", 0.0)


        # NEW: derived — speed deficit
        self.xpos_norm = self.norm_pos(self.xpos, self.min_x, self.max_x)
        self.ypos_norm = self.norm_pos(self.ypos, self.min_y, self.max_y)
        self.zpos_norm = self.norm_pos(self.zpos, self.min_z, self.max_z)
        
        # NEW: airborne / surface
        self.frames_in_air = float(state.get("frames_in_air", 0))
        self.pitch = state.get("pitch", 0.0)
        self.surf_norm_x = state.get("surf_norm_x", 0.0)
        self.surf_norm_y = state.get("surf_norm_y", 1.0)
        self.surf_norm_z = state.get("surf_norm_z", 0.0)

        # NEW: speed modifiers
        self.offroad_speed = state.get("offroad_speed", 1.0)
        self.effect_speed = state.get("effect_speed", 1.0)
        self.air_speed = state.get("air_speed", 1.0)
        self.boost_mt = state.get("boost_mt", 0) / 63   # 63 being maxed mini turbo frames
        self.max_speed_fraction = state.get("max_speed_fraction", 1.0)

        # NEW: respawn
        self.prb_flag = int(state.get("prb_flag", 0))

        # NEW: derived — heading rate (angular velocity of facing angle)
        self.heading_rate = ((self.angle_facing - self._prev_angle_facing + 1) % 2) - 1
        self._prev_angle_facing = self.angle_facing

        # NEW: derived — speed deficit
        max_s = self.max_speed if self.max_speed > 0 else 1
        raw_speed = state.get("speed", 0) * max_s   # reverse the /max_speed normalisation
        self.speed_deficit = float(numpy.clip((max_s - raw_speed) / max_s, 0.0, 1.0))
        
        # Drift_status
        was_drifting = self._prev_drift_direction != 0
        is_drifting = self.drift_direction != 0
        self.just_released_drift = was_drifting and not is_drifting
        # Update prev_drift_dir
        self._prev_drift_direction = self.drift_direction

        
        # ------------------------------------------------------------------ #
        # NEW: Horizon curvature lookup
        # ------------------------------------------------------------------ #
        n_pts = len(self.centerline.new_sin)
        idx_short = (self.centerline_seg_idx + 600) % n_pts
        idx_mid   = (self.centerline_seg_idx + 1200) % n_pts
        idx_long  = (self.centerline_seg_idx + 1800) % n_pts
        
        # Compare current track direction to future track direction
        # Positive = track curves left, Negative = track curves right
        self.horizon_short = float(self.centerline.angle_to_centerline(
            self.current_cp_sin, self.current_cp_cos,
            self.centerline.new_sin[idx_short], self.centerline.new_cos[idx_short]
        ))
        
        self.horizon_mid = float(self.centerline.angle_to_centerline(
            self.current_cp_sin, self.current_cp_cos,
            self.centerline.new_sin[idx_mid], self.centerline.new_cos[idx_mid]
        ))
        
        self.horizon_long = float(self.centerline.angle_to_centerline(
            self.current_cp_sin, self.current_cp_cos,
            self.centerline.new_sin[idx_long], self.centerline.new_cos[idx_long]
        ))

    # ---------------------------------------------------------------------- #
    # FORMAT OBSERVATION
    # ---------------------------------------------------------------------- #
    def format_observation(self, state, isreset):
        is_collision = numpy.array([0 if isreset else self.is_collision], dtype=numpy.float32)
        _vel_norm = max(self.max_kart_speed_default / 4096.0, 1e-9)

        return {
            # Existing
            "speed":            numpy.array([self.speed],      dtype=numpy.float32),
            "angle":            numpy.array([self.angle],           dtype=numpy.float32),
            "angle_facing":     numpy.array([self.angle_facing],    dtype=numpy.float32),
            "drift_angle":      numpy.array([self.drift_angle],     dtype=numpy.float32),
            "centerline_dist":  numpy.array([self.centerline_dist], dtype=numpy.float32),
            "is_collision":     is_collision,
            "grip":             numpy.array([self.grip],            dtype=numpy.float32),
            "drift_direction":  numpy.array([self.drift_direction], dtype=numpy.float32),
            "drift_progress":   numpy.array([self.drift_progress],  dtype=numpy.float32),
            "mttime":           numpy.array([self.mttime],          dtype=numpy.float32),
            "mt_left":          numpy.array([self.mt_left],         dtype=numpy.float32),
            "mt_right":         numpy.array([self.mt_right],        dtype=numpy.float32),
            "air":              numpy.array([self.air],             dtype=numpy.float32),
            "trackstatus":      numpy.array([self.trackstatus],     dtype=numpy.float32),
            "boost_mt":                 numpy.array([self.boost_mt], dtype=numpy.float32),
            "heading_rate":         numpy.array([self.heading_rate], dtype=numpy.float32),
            "pitch":                 numpy.array([self.pitch], dtype=numpy.float32),
            "prb_flag":             numpy.array([self.prb_flag], dtype=numpy.float32),
            "slip_angle":             numpy.array([self.slip_angle], dtype=numpy.float32),
            
            
            "surf_norm_x":      numpy.array([self.surf_norm_x],     dtype=numpy.float32),
            "surf_norm_y":      numpy.array([self.surf_norm_y],     dtype=numpy.float32),
            "surf_norm_z":      numpy.array([self.surf_norm_z],     dtype=numpy.float32),
            
            "xpos_norm":      numpy.array([self.xpos_norm],     dtype=numpy.float32),
            "ypos_norm":      numpy.array([self.ypos_norm],     dtype=numpy.float32),
            "zpos_norm":      numpy.array([self.zpos_norm],     dtype=numpy.float32),
            
            "vel_x": numpy.array([numpy.clip(self.vel_x / _vel_norm, -1.0, 1.0)], dtype=numpy.float32),
            "vel_y": numpy.array([numpy.clip(self.vel_y / _vel_norm, -1.0, 1.0)], dtype=numpy.float32),
            "vel_z": numpy.array([numpy.clip(self.vel_z / _vel_norm, -1.0, 1.0)], dtype=numpy.float32),
            "vertical_velocity": numpy.array([numpy.clip(self.vertical_velocity / _vel_norm, -1.0, 1.0)], dtype=numpy.float32),

            "offroad_speed":  numpy.array([numpy.clip(self.offroad_speed  - 1.0, -1.0, 1.0)], dtype=numpy.float32),
            "effect_speed":   numpy.array([numpy.clip(self.effect_speed   - 1.0, -1.0, 1.0)], dtype=numpy.float32),
            "air_speed":      numpy.array([numpy.clip(self.air_speed      - 1.0, -1.0, 1.0)], dtype=numpy.float32),
                        # max_speed_fraction: already ~[0,1], safe to keep, clip just in case
            "max_speed_fraction": numpy.array([numpy.clip(self.max_speed_fraction, 0.0, 1.0)], dtype=numpy.float32),
            "speed_deficit":    numpy.array([self.speed_deficit],      dtype=numpy.float32),
            # Horizons
            "horizon_short":    numpy.array([self.horizon_short],      dtype=numpy.float32),
            "horizon_mid":      numpy.array([self.horizon_mid],        dtype=numpy.float32),
            "horizon_long":     numpy.array([self.horizon_long],       dtype=numpy.float32),
            
            "obsppm" :             numpy.array([self.obsppm],       dtype=numpy.float32),
            }

    # ---------------------------------------------------------------------- #
    # RESET ENVIRONMENT — zero / default all new variables too
    # ---------------------------------------------------------------------- #
    def reset_environment(self):
        # Existing
        self.checkpoints = 0
        self.lap_f = 0
        self.max_speed = 0
        self.speed = 0
        self.xpos = 0
        self.ypos = 0
        self.zpos = 0
        self.angle_facing = 0
        self.drift_angle = 0
        self.sum_angle = 0
        self.current_cp_angle = 0
        self.current_cp_sin = 0
        self.current_cp_cos = 0
        self.grip = 0
        self.turn_loss = 0
        self.air = 0
        self.mttime = 0
        self.mt_left = 0
        self.mt_right = 0
        self.drift_direction = 0
        self.drift_progress = 0
        self.boost_timer = 0
        self.angle = 0
        self.real_speed = 0
        self.trackstatus = 0
        self.positions = deque([(0, 0, 0)] * 2, maxlen=2)
        self.position = (0, 0, 0)
        self.wallclip = 1
        
        self.xpos_norm = 0
        self.ypos_norm = 0
        self.zpos_norm = 0

        # NEW — reset to safe defaults (grounded, flat surface, full traction)
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.vel_z = 0.0
        self.vertical_velocity = 0.0
        self.slip_angle = 0.0
        self.frames_in_air = 0.0
        self.pitch = 0.0
        self.surf_norm_x = 0.0
        self.surf_norm_y = 1.0      # flat ground default
        self.surf_norm_z = 0.0
        self.offroad_speed = 1.0
        self.effect_speed = 1.0
        self.air_speed = 1.0
        self.boost_mt = 0.0
        self.max_speed_fraction = 1.0
        self.prb_flag = 0
        self.heading_rate = 0.0
        self.speed_deficit = 0.0
        self._prev_angle_facing = 0.0

        # Centerline
        self.centerline_dist = 0.0
        self.centerline_seg_idx = 0
        self.progress_float = 0.0
        self.prev_progress_float = 0.0

        self.reset_emulator()
        self.frame_count = 0
        self.current_checkpoint = 0
        self.next_checkpoint = 1
        self.total_reward = 0
        self.done = False
        self.speed_queue = deque([0] * self.frame_buffer_size, maxlen=self.frame_buffer_size)
        self.collision_n = 0
        self.max_kart_speed_default = self.max_speed
        self.is_collision = 0
        
        self.speed_deficit = 0.0
        self._prev_angle_facing = 0.0
        
        self.horizon_short = 0.0
        self.horizon_mid = 0.0
        self.horizon_long = 0.0
        self.obsppm = 0.0

    # ---------------------------------------------------------------------- #
    # STEP (unchanged logic, kept for completeness)
    # ---------------------------------------------------------------------- #
    def step(self, action):
        self.apply_action(action)
        for _ in range(3):
                    
                    if self.isrender == 'human':
                        self.emu.cycle()
                    else:
                        
                        self.emu.cycle_frameskip(with_joystick=False) 
        
        state = self.get_emulator_state(self.angle_vals)



        
        self.real_speed = self.compute_real_speed(state)
        self.update_state(state)

        if self.mttime > 0:
            self.metrics.set_flag(self.emu, 0x217ACF8, 0x4B, 0x20)
            self.prb_flag = 1

        self.trackstatus_queue.append(self.trackstatus)
        self.total_distance_queue.append(self.total_distance)

        self.delta_progress = self.progress_float - self.prev_progress_float
        self.delta_progress = numpy.clip(self.delta_progress, -2.0, 5.0)

        self.centerline_seg_queue.append(self.centerline_seg_idx)
        self.positions.append(self.position)

        try:
            tmp1 = self.centerline_seg_queue[1] + 20000 * (self.trackstatus_queue[1] // 0.33)
            tmp2 = self.centerline_seg_queue[0] + 20000 * (self.trackstatus_queue[0] // 0.33)
            delta = tmp1 - tmp2
            if delta > 10000:
                delta -= 20000
            elif delta < -10000:
                delta += 20000
            self.ppm = delta
        except ZeroDivisionError:
            self.ppm = 0
        
        self.obsppm = self.ppm /42
        self.speed_queue.append(self.speed)
        self.update_collision_state()
        self.mt_cooldown = max(0, self.mt_cooldown - 1)

        reward = self.calculate_reward()
        self.total_reward += reward
        self.frame_count += 1

        self.prev_progress_float = self.progress_float
        self.check_checkpoint()

        if self.isrender == 'human':
            self.window.draw()
    
        observation = self.format_observation(state, isreset=False)

        if self.frame_count % 10 == 0:
            print(f"[{self.savestate_number}] Frame {self.frame_count}: "
                  f"Reward = {reward:.3f}, Total = {self.total_reward:.3f}, "
                  f"CP = {self.checkpoints}, angle = {self.angle:.3f},",
                  f"mt = {self.mttime}, trk = {self.trackstatus:.3f}")
            
            # print(self.xpos_norm, self.ypos_norm, self.zpos_norm)
        
        # updating dict for callbacks report
        info = dict(state)
        info["ppm"] = float(self.ppm)
        info["is_collision"] = int(self.is_collision)
        info["boost_mt"] = float(self.boost_mt)
        info["drift_direction"] = float(self.drift_direction)
        info["trackstatus"] = float(self.trackstatus)
        info["lap_f"] = float(self.lap_f)  
        
        
        
        self.done = self.check_done_condition()
        
        if self.done and self.done_reason == "timeout":
            info["TimeLimit.truncated"] = True
        
        
        return observation, reward, self.done, info

    def reset(self):
        with self.utils.suppress_stdout_stderr():
            self.reset_emulator()

        state = self.get_emulator_state(self.angle_vals)
        self.update_state(state)
        observation = self.format_observation(state, isreset=True)

        self.total_reward = 0
        self.collision_n = 0
        self.frame_count = 0
        self.done = False
        self.max_kart_speed_default = self.max_speed
        self.mt_cooldown = 0
        self.mt_left = 0
        self.mt_right = 0
        self._prev_angle_facing = self.angle_facing   # seed heading_rate to avoid spike at step 1
        

        if self.savestate_number > 1:
            self.speed_queue = deque([state.get("speed")] * 5, maxlen=5)
            self.trackstatus_queue = deque([state.get("trackstatus")] * 2, maxlen=2)
            self.centerline_seg_queue = deque([self.centerline_seg_idx] * 2, maxlen=2)
            self.positions = deque([self.position] * 2, maxlen=2)

        if self.savestate_number == 1:
            self.speed_queue = deque([0] * 5, maxlen=5)
            self.trackstatus_queue = deque([0] * 2, maxlen=2)
            self.current_checkpoint = 0
            self.next_checkpoint = 1
            self.trackstatus = 0
            self.centerline_seg_queue = deque([0] * 2, maxlen=2)
            self.positions = deque([self.position] * 2, maxlen=2)

        self.emu.memory.write_long(0x220BC244, 0xE3510000)
        self.emu.memory.write_long(0x220BA6B8, 0XE3500000)

        return observation

    # ---------------------------------------------------------------------- #
    # UNCHANGED helpers
    # ---------------------------------------------------------------------- #


    def get_emulator_state(self, angle_vals):
        return self.metrics.get_values(self.emu, self.angle_vals)



    def calculate_reward(self):
        
        # ---- Reward breakdown ---- #
        # ppm: 20 000 per lap. Given 30sec per step at 20 fps
        # it should sit around 30/step
        # NB --> if reward < -100 or reward > 100: dirty fix for start line
        # 
        # boost_mt | 2.0 | reward if mini turbo
        # snaking: | 1.0 | reward for turbo chaining
        # drifting | 0.1 | reward for drifting, needed for early mt learning
        
        # Special
        # trackstatus | +x.x | reward for completing a lap. the faster the higher 
        
        
        # Penalties
        # just_released_drift | -0.5 | cancel drift
        # wallclip            | -1.0 | collisions

        
        # reward = - 0.5 # penalty for existing
        reward=-1
        reward = reward + (self.ppm * self.speed) / 30 # expected around 1, ideally higher
       
        
        if reward < -100 or reward > 100:
            reward = 0
            print("CLIPPED")
        if self.boost_mt > 0:
            reward += 0.5
            print(" --- MT ---")
        if self.mt_cooldown == 0 and self.boost_mt > 0 and self.drift_progress > 0.99:
            reward += 1
            self.mt_cooldown = 10
            print(" SNAKING ---")
            
        if (abs(self.turn_loss) > 0.5) and self.drift_direction == 0:
            reward -= 0.5
            
        if self.drift_direction != 0:
            reward += 0.2
     
            
        if self.just_released_drift and not self.boost_mt > 0:
            reward = reward - 1
            # print("Released")

        # if self.grip < 0.5:
        #     reward -= 0.5
        
        if self.wallclip < 1:
            reward -= 3
        
        # if self.trackstatus > 0.999:
        #     reward = reward +( 1800 - self.frame_count) 
        #     print(f"FINISHED IN {self.frame_count/20}")
        
        return reward
    
    # def calculate_reward_tuning(self):
    #     # 1. Base living penalty to encourage finding the fastest line
    #     reward = -0.2 
        
    #     # 2. Prevent negative squaring if the car is stuck/backward
    #     safe_ppm = max(0.0, float(self.ppm))
        
    #     # 3. Quadratic Scaling around your current 34 PPM plateau.
    #     # At 34 PPM -> (34/34)^2 = 1.0 reward
    #     # At 40 PPM -> (40/34)^2 = 1.38 reward (Massive incentive for WR speed)
    #     # At 20 PPM -> (20/34)^2 = 0.34 reward (Harsh drop-off for slow corners)
    #     ppm_reward = (safe_ppm / 37.0) ** 2
        
    #     reward += ppm_reward
        
    #     if self.prb_flag==1:
    #         reward = reward + 0.2
        
    #     if self.prb_flag==0:
    #         reward = reward - 1

    #     #if self.prb_flag==0:
    #     #    reward =reward - 3

    #     if self.just_released_drift and not self.boost_mt > 0:
    #         reward = reward - 0.5
    
    #     # 4. Minimal wall penalty. 
    #     # We keep this so it doesn't wall-bang, but we don't make it -10 
    #     # so the AI is still willing to risk getting extremely close to the wall.
    #     if self.wallclip < 1:
    #         reward -= 1.0
    #     if self.boost_mt==0:
    #         reward = reward - 0.1
    
    #     # 5. Clip the final reward to keep PPO perfectly stable
    #     reward = float(numpy.clip(reward, -2.0, 3.0))
        
    #     return reward

        
    def check_done_condition(self):
        # 1. Did we run out of time? (Truncation)
        if self.frame_count >= self.episodelength:
            self.done_reason = "timeout"
            return True
            
        # 2. Did we win? (Termination)
        if self.trackstatus > 0.999:
            self.done_reason = "finish"
            return True
        

            
        # 3. Did we crash? (Termination)
        if self.is_collision and self.early_term == 'true':
            self.done_reason = "crash"
            return True
            
        self.done_reason = "none"
        return False
    

    def apply_action(self, action):
        move_functions = {
            0: self.moves.move_forward,
            1: self.moves.move_left,
            2: self.moves.move_right,
            3: self.moves.press_drift,
            4: self.moves.release_drift,
        }
        move_functions[action](self.emu)

    def compute_real_speed(self, state):
        real_speed = math.sqrt(
            (self.xpos - state["xpos"]) ** 2 + (self.ypos - state["ypos"]) ** 2)
        return real_speed / (3 * self.max_kart_speed_default)

    def update_collision_state(self):
        mean_speed = numpy.mean(self.speed_queue)
        self.is_collision = int(self.speed < 0.75 * mean_speed)
        self.speed_queue.append(self.speed)

    def check_checkpoint(self):
        if self.checkpoints == self.current_checkpoint + 1:
            self.current_checkpoint += 1
            self.next_checkpoint += 1
            if self.next_checkpoint == 26:
                self.next_checkpoint = 0
                
    def norm_pos(self,x,min,max):
        min = min - abs((min * 0.05))
        max = max + abs((max *0.05))
        normed = 2 * ((x-min) / (max-min)) - 1 
        return(normed)
        
        
