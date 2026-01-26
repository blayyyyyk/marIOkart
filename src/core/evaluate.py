import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import gymnasium as gym
from core.env import MKDS_DeSmuME_Env
from torch._prims_common import DeviceLikeType
from typing import Optional, Any, Callable
from models.lstm import MarioKartLSTM, CONFIG, KEYMAP
from src.main import EmulatorWindow, OVERLAYS, start_keyboard_listener, USER_KEYMAP
from desmume.controls import keymask
import gi
gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gtk, Gdk, GLib

def _user_input_catch(func: Callable):
    try:
        val = True
        while val:
            val = func()
        return True
    except KeyboardInterrupt:
        return False


# evaluate mariokart ds agent in gymnasium environment
def evaluate_agent(
    model: nn.Module, # model instance
    checkpoint_path: str, # path of the saved model weights
    rom_path: str, # path to emulator rom
    state_path: str, # path to desmume save state to start evaluation from
    history_size: int = 10, # size of lstm memory
    keymap: dict | list = KEYMAP, # mapping of integers to keymask
    log_interval: int = 60, # number of frames between logging stats to console
    device: DeviceLikeType = "cpu" # tensor storage device
):
    env = MKDS_DeSmuME_Env(rom_path, state_path, keymap, history_size, render_mode="human")
    env.emu.volume_set(0)

    device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    window = EmulatorWindow(env.emu, OVERLAYS, device)

    print("Model loaded.")
    print("Starting evaluation...")

    obs, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0
    step_count = 0
    listener, input_state = start_keyboard_listener()

    def _loop():
        nonlocal total_reward, step_count, obs, info, terminated, truncated
        wall_dists = obs['wall_distances'].unsqueeze(0).to(device)
        prev_actions = obs['actions'].unsqueeze(0).to(device)

        # inference
        with torch.no_grad():
            
            logits = model(wall_dists, prev_actions)
            action = torch.argmax(logits, dim=1).item()

        # interaction
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

        if step_count % log_interval == 0:
            print(f"Step {step_count} | Reward: {total_reward:.4f} | Action: {action}")

        if terminated or truncated:
            window.quit()

        return True

    def _tick():
        for key in input_state:
            env.emu.input.keypad_add_key(keymask(USER_KEYMAP[key]))
        
        if env.emu.input.keypad_get() != 0:
            env.emu.cycle()

        window.update(env.emu)
        return True
    
    GLib.timeout_add(16, _tick)
    GLib.timeout_add(60, _loop)
    Gtk.main()
    window.connect("destroy", lambda w: window.worker_queue.put(None))

    print(f"Episode finished. Total reward: {total_reward}")


    
# entry function
def main():
    model = MarioKartLSTM(
        CONFIG['obs_dim'], 
        CONFIG['num_classes'], 
        CONFIG['hidden_dim'], 
        CONFIG['embed_dim'],
    )

    rom_path = "private/mariokart_ds.nds"
    state_path = "private/state_slot_0.dst"
    checkpoint_path = "private/checkpoints/checkpoint_e01122026.pth"
    evaluate_agent(model, checkpoint_path, rom_path, state_path)

if __name__ == "__main__":
    main()