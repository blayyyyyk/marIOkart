from gym_mkds.wrappers.sweeping_ray import get_standing_triangle_id
from src.environments.checkpoint_wrapper import checkpoint_angle_signed
from typing import cast
import gymnasium as gym
import numpy as np
from desmume.emulator_mkds import MarioKart
from src.environments.boundary_wrapper import project_2d


class AdvancedObservations(gym.ObservationWrapper):
    def __init__(self, env: gym.Env[gym.spaces.Dict, gym.spaces.Discrete]):
        super().__init__(env)
        
        self.observation_space = gym.spaces.Dict({
            # --- Existing ---
            "speed":            gym.spaces.Box(low=0,    high=1.3, shape=(1,), dtype=np.float32),
            "angle":            gym.spaces.Box(low=-1,   high=1,   shape=(1,), dtype=np.float32),
            "angle_facing":     gym.spaces.Box(low=-1,   high=1,   shape=(1,), dtype=np.float32),
            "drift_angle":      gym.spaces.Box(low=-1,   high=1,   shape=(1,), dtype=np.float32),
            "is_collision":     gym.spaces.Box(low=0,    high=1,   shape=(1,), dtype=np.float32),
            "centerline_dist":  gym.spaces.Box(low=-1,   high=1,   shape=(1,), dtype=np.float32),
            "grip":             gym.spaces.Box(low=0,    high=1,   shape=(1,), dtype=np.float32),
            "drift_direction":  gym.spaces.Box(low=-1,   high=1,   shape=(1,), dtype=np.float32),
            "drift_progress":   gym.spaces.Box(low=0,    high=1,   shape=(1,), dtype=np.float32),
            "mt_left":          gym.spaces.Box(low=0,    high=1,   shape=(1,), dtype=np.float32),
            "mt_right":         gym.spaces.Box(low=0,    high=1,   shape=(1,), dtype=np.float32),
            "mttime":           gym.spaces.Box(low=0,    high=1,   shape=(1,), dtype=np.float32),
            "air":              gym.spaces.Box(low=0,    high=1,   shape=(1,), dtype=np.float32),
            "trackstatus":      gym.spaces.Box(low=0,    high=1,   shape=(1,), dtype=np.float32),
    
            "xpos_norm":  gym.spaces.Box(low=-1,   high=1,   shape=(1,), dtype=np.float32),
            "ypos_norm":  gym.spaces.Box(low=-1,   high=1,   shape=(1,), dtype=np.float32),
            "zpos_norm":  gym.spaces.Box(low=-1,   high=1,   shape=(1,), dtype=np.float32),
    
            # --- NEW: velocity vector ---
            # Actual per-frame displacement in kart-local coords, fixed-point /4096.
            "vel_x":             gym.spaces.Box(low=-1, high=1,  shape=(1,), dtype=np.float32),
            "vel_y":             gym.spaces.Box(low=-1, high=1,  shape=(1,), dtype=np.float32),
            "vel_z":             gym.spaces.Box(low=-1, high=1,  shape=(1,), dtype=np.float32),
            "vertical_velocity": gym.spaces.Box(low=-1, high=1,  shape=(1,), dtype=np.float32),
    
            # Slip angle: facing direction vs actual movement direction, [-1,1]
            "slip_angle":       gym.spaces.Box(low=-1,   high=1,   shape=(1,), dtype=np.float32),
    
            # Nose up/down angle, same normalisation as angle_facing.
            "pitch":            gym.spaces.Box(low=-1,   high=1,   shape=(1,), dtype=np.float32),
            # Surface normal XYZ: encodes road slope and banking.
            "surf_norm_x":      gym.spaces.Box(low=-1,   high=1,   shape=(1,), dtype=np.float32),
            "surf_norm_y":      gym.spaces.Box(low=-1,   high=1,   shape=(1,), dtype=np.float32),
            "surf_norm_z":      gym.spaces.Box(low=-1,   high=1,   shape=(1,), dtype=np.float32),
            # --- NEW: speed modifiers ---
            # offroad < 1 means grass/sand penalty; effect > 1 means item boost.
            "offroad_speed":     gym.spaces.Box(low=-1, high=1,  shape=(1,), dtype=np.float32),
    
            "effect_speed":      gym.spaces.Box(low=-1, high=1,  shape=(1,), dtype=np.float32),
            "air_speed":         gym.spaces.Box(low=-1, high=1,  shape=(1,), dtype=np.float32),
    
            # Binary: 1 = MT boost currently firing.
            "boost_mt":         gym.spaces.Box(low=0,    high=1,   shape=(1,), dtype=np.float32),
    
            # Fraction of max speed currently allowed by engine [0, ~1.3].
            "max_speed_fraction":gym.spaces.Box(low=0,  high=1,  shape=(1,), dtype=np.float32),
    
            # --- PRB ---
            "prb_flag":         gym.spaces.Box(low=0,    high=1,   shape=(1,), dtype=np.float32),
            # --- NEW: derived ---
            "heading_rate":     gym.spaces.Box(low=-1,   high=1,   shape=(1,), dtype=np.float32),
            "speed_deficit":    gym.spaces.Box(low=0,    high=1,   shape=(1,), dtype=np.float32),
            # --- NEW: track horizon / curvature ---
            "horizon_short":    gym.spaces.Box(low=-1,   high=1,   shape=(1,), dtype=np.float32),
            "horizon_mid":      gym.spaces.Box(low=-1,   high=1,   shape=(1,), dtype=np.float32),
            "horizon_long":     gym.spaces.Box(low=-1,   high=1,   shape=(1,), dtype=np.float32),
            "obsppm" :             gym.spaces.Box(low=-1,   high=1,   shape=(1,), dtype=np.float32),
        
        })
        
    def _centerline_dist(self):
        emu: MarioKart = cast(MarioKart, self.get_wrapper_attr('emu'))
        if not emu.memory.race_ready:
            return 0.0
            
        p0, p1 = np.unstack(emu.memory.checkpoint_pos()["current_checkpoint_pos"], axis=0)
        mid0 = (p0 + p1) / 2
        
        p2, p3 = np.unstack(emu.memory.checkpoint_pos()["next_checkpoint_pos"], axis=0)
        mid1 = (p2 + p3) / 2
        
        kart_position = emu.memory.driver_position
        centerline_intersect = project_2d(mid0[None, :], mid1[None, :], kart_position)
        centerline_dist = np.linalg.norm(centerline_intersect - kart_position)
        return centerline_dist
        
    def _get_obs(self):
        emu: MarioKart = cast(MarioKart, self.get_wrapper_attr('emu'))
        if not emu.memory.race_ready:
            assert isinstance(self.observation_space, gym.spaces.Dict), "wrong observation space type"
            return { k: 0.0 for k in self.observation_space.keys() }
            
        out = {}
        
        out["speed"] = float(emu.memory.driver.speed)
        out["angle"] = checkpoint_angle_signed(emu, direction_mode="movement")
        out["angle_facing"] = checkpoint_angle_signed(emu, direction_mode="direction")
        out["drift_angle"] = abs(out["angle"]) - abs(out["angle_facing"])
        out["is_collision"] = 0.0 # TODO
        out["centerline_dist"] = self._centerline_dist()
        out["grip"] = 0.0 # TODO
        out["drift_direction"] = 0.0 # emu.memory.driver_velocity?
        out["drift_progress"] = 0.0 #TODO
        out["mt_left"] = 0.0 # TODO
        out["mt_right"] = 0.0 # TODO
        out["mttime"] = 0.0 # TODO
        out["air"] = 0.0 # TODO
        out["trackstatus"] = 0.0 # TODO
        out["xpos_norm"] = 0.0 # TODO
        out["ypos_norm"] = 0.0 # TODO
        out["zpos_norm"] = 0.0 # TODO
        out["vel_y"] = 0.0 # TODO
        out["vel_z"] = 0.0 # TODO
        out["vertical_velocity"] = 0.0 # TODO
        out["slip_angle"] = 0.0 # TODO
        out["pitch"] = 0.0 # TODO
        out["surf_norm_x"] = 0.0 # TODO
        out["surf_norm_y"] = 0.0 # TODO
        out["surf_norm_z"] = 0.0 # TODO
        out["offroad_speed"] = 0.0 # TODO
        out["effect_speed"] = 0.0 # TODO
        out["air_speed"] = 0.0 # TODO
        out["boost_mt"] = 0.0 # TODO
        out["max_speed_fraction"] = 0.0 # TODO
        out["prb_flag"] = 0.0 # TODO
        out["heading_rate"] = 0.0 # TODO
        out["speed_deficit"] = 0.0 # TODO
        out["horizon_short"] = 0.0 # TODO
        out["horizon_mid"] = 0.0 # TODO
        out["horizon_long"] = 0.0 # TODO
        out["obsppm"] = 0.0 # TODO
        
        return out
        
        
    def observation(self, observation):
        return self._get_obs()
        