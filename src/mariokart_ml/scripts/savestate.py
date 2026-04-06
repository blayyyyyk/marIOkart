import gymnasium as gym
import numpy as np
from desmume.emulator_mkds import MarioKart
from gym_mkds.wrappers import GtkWindow, HumanInput
from gym_mkds.wrappers.window import EnvWindow
from gymnasium.wrappers import FrameStackObservation, ReshapeObservation

from ..environments import *

padding_config = {
    'wall_distances': (20,),        # Always pad to shape (10, 3)
    'checkpoint_angle': (1,)  # Always pad to shape (5, 2)
}



def main():
    env = gym.make("gym_mkds/MarioKartDS-human-v1")
    env = HumanInput(env)
    env = FrameStackObservation(env, stack_size=200, padding_type="zero")
    env = GtkWindow(env, scale=2)
    obs, info = env.reset()
    print(obs)

    saved = False
    assert env.window is not None
    while env.window.is_alive:
        emu: MarioKart = env.get_wrapper_attr('emu')
        actions = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(actions)
        if emu.memory.race_ready and not saved:
            emu.savestate.save_file(f"Course_{emu.memory.stag_data.courseId}.dst")
            saved = True

if __name__ == "__main__":
    main()
